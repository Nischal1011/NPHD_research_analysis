import requests
import pandas as pd
import numpy as np
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class CensusAPIWrapper:
    BASE_URL = "https://api.census.gov/data/2023/acs/acs5"

    def __init__(self, api_key):
        self.api_key = api_key

    def get_tract_data(self, variables):
        """Fetch ACS data at census tract level for all states, handling >50 variables."""
        all_data = []
        states = self.get_states()
        
        # Split variables into chunks of 50 or fewer
        chunk_size = 50 - 1  # Reserve 1 spot for "NAME"
        variable_chunks = [variables[i:i + chunk_size] for i in range(0, len(variables), chunk_size)]
        
        for state_fips in states:
            state_data = []
            for chunk in variable_chunks:
                params = {
                    "get": ",".join(chunk + ["NAME"]),
                    "for": "tract:*",
                    "in": f"state:{state_fips}",
                    "key": self.api_key
                }
                df = self._make_request(params)
                state_data.append(df)
            # Merge data for this state across chunks
            if state_data:
                merged_state_df = state_data[0]
                for df in state_data[1:]:
                    merged_state_df = merged_state_df.merge(
                        df, on=["NAME", "state_fips", "county_fips", "tract_fips"], how="outer"
                    )
                all_data.append(merged_state_df)
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def get_states(self):
        """Retrieve FIPS codes for all U.S. states."""
        geo_url = "https://api.census.gov/data/2023/acs/acs5"
        params = {"get": "NAME", "for": "state:*", "key": self.api_key}
        response = requests.get(geo_url, params=params)
        
        if response.status_code != 200:
            print(f"Failed to fetch states: {response.status_code}")
            print(f"Response text: {response.text}")
            response.raise_for_status()
        
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Invalid JSON response: {response.text}")
            raise
        
        return [row[1] for row in data[1:]]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.JSONDecodeError))
    )
    def _make_request(self, params):
        """Handle API requests with retry and return a DataFrame."""
        response = requests.get(self.BASE_URL, params=params)
        
        if response.status_code != 200:
            print(f"API Request Failed: {response.status_code}")
            print(f"Response text: {response.text}")
            response.raise_for_status()
        
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON. Response text: {response.text}")
            raise
        
        if not data or len(data) < 2:
            print(f"No data returned for params: {params}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data[1:], columns=data[0])
        df = df.rename(columns={"tract": "tract_fips", "state": "state_fips", "county": "county_fips"})
        return df

if __name__ == "__main__":
    API_KEY = os.getenv("census_key")
    if not API_KEY:
        raise ValueError("Please set the 'census_key' environment variable with your Census API key.")
    census_api = CensusAPIWrapper(API_KEY)

    # Original variables
    elderly_vars = [f'B01001_{str(i).zfill(3)}E' for i in range(20, 26)] + \
                   [f'B01001_{str(i).zfill(3)}E' for i in range(44, 50)]
    disabled_vars = [f'B18101_{str(i).zfill(3)}E' for i in [4, 7, 10, 13, 16, 19, 23, 26, 29, 32, 35, 38]]
    cost_burden_vars = [f'B25070_{str(i).zfill(3)}E' for i in range(7, 11)] + ['B25070_001E']
    total_pop_var = ['B01003_001E']

    # Additional variables
    additional_vars = [
        'B25077_001E',  # Median home value
        'B25064_001E',  # Median gross rent
        'B25003_002E',  # Owner-occupied units
        'B25003_003E',  # Renter-occupied units
        'B25004_001E',  # Vacant units
        'B19013_001E',  # Median household income
        'B15003_022E',  # Bachelor's degree (25+)
        'B01001_001E',  # Total population (already in total_pop_var)
        'B23025_005E',  # Unemployed in labor force
        'B19083_001E',  # Gini index
        'B02001_002E',  # White alone
        'B02001_003E',  # Black alone
        'B03001_003E',  # Hispanic or Latino
        'B02001_005E',  # Asian alone
        'B07001_017E',  # Same residence 1 year
        'B07001_033E',  # Moved within county
        'B07001_049E',  # Moved from different county
        'B07003_004E',  # Owner tenure (recent movers)
        'B07003_010E',  # Renter tenure (recent movers)
        'B11001_001E',  # Total households
        'B11001_002E',  # Family households
        'B11001_007E',  # Households with kids
        'B01001_006E',  # Under 18 population
        'B25034_010E', 'B25034_009E', 'B25034_008E', 'B25034_007E',  # Built years
        'B25034_006E', 'B25034_005E', 'B25034_004E', 'B25034_003E', 'B25034_002E'
    ]

    # Combine all variables (avoid duplicates)
    variable_list = list(set(total_pop_var + elderly_vars + disabled_vars + cost_burden_vars + additional_vars))
    print(f"Total variables to fetch: {len(variable_list)}")

    print("Fetching ACS data for all census tracts across all states...")
    df_acs = census_api.get_tract_data(variable_list)

    numeric_cols = [col for col in df_acs.columns if col not in ["NAME", "state_fips", "county_fips", "tract_fips"]]
    df_acs[numeric_cols] = df_acs[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Original derived columns
    df_acs['Population_65_and_over'] = df_acs[elderly_vars].sum(axis=1)
    df_acs['Population_with_disability'] = df_acs[disabled_vars].sum(axis=1)
    df_acs['pct_cost_burdened_renters'] = (df_acs[[f'B25070_{str(i).zfill(3)}E' for i in range(7, 11)]]
                                           .sum(axis=1) / df_acs['B25070_001E'].replace(0, np.nan)) * 100

    df_acs['CensusTract'] = df_acs['state_fips'] + df_acs['county_fips'] + df_acs['tract_fips']

    output_path = "data/tract_census_data_all_states_v1.csv"
    os.makedirs("data", exist_ok=True)
    df_acs.to_csv(output_path, index=False)
    print(f"ACS data for all census tracts saved to '{output_path}'")