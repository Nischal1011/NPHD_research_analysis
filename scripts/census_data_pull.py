import censusdis.data as ced
import geopandas as gpd
import pandas as pd
from censusdis.datasets import ACS5
from censusdis import states
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import retrying

def download_state_data(state_fips, dataset, vintage, variables, max_retries=3, retry_delay=5):
    """Download data for a single state with retries, returning a GeoDataFrame or None on error."""
    state_name = state_fips  # You can map FIPS to state names if needed
    print(f"--- Downloading data for: {state_name} ---")

    @retrying.retry(
        stop_max_attempt_number=max_retries,
        wait_fixed=retry_delay * 1000,  # Convert seconds to milliseconds
        retry_on_exception=lambda e: isinstance(e, Exception)  # Retry on any exception
    )
    def attempt_download():
        try:
            start_time = time.time()
            gdf_tracts = ced.download(
                dataset=dataset,
                vintage=vintage,
                download_variables=variables,
                state=state_fips,
                tract='*',
                with_geometry=True,
            )
            print(f"Downloaded {len(gdf_tracts)} tracts for {state_name} in {time.time() - start_time:.2f} seconds.")
            return gdf_tracts
        except Exception as e:
            print(f"Error downloading data for {state_name}: {e}")
            raise  # Re-raise the exception to trigger retry

    try:
        return attempt_download()
    except Exception as e:
        print(f"Failed to download data for {state_name} after {max_retries} attempts: {e}")
        return None

if __name__ == "__main__":
    # --- Configuration ---
    ACS_VINTAGE = 2023
    OUTPUT_FILENAME = f"data/acs5_{ACS_VINTAGE}_tract_v1_data_multi_state.parquet"

    # Define the list of states
    state_fips_list = [
        states.AL, states.AK, states.AZ, states.AR, states.CA, states.CO, states.CT,
        states.DE, states.DC, states.FL, states.GA, states.HI, states.ID, states.IL,
        states.IN, states.IA, states.KS, states.KY, states.LA, states.ME, states.MD,
        states.MA, states.MI, states.MN, states.MS, states.MO, states.MT, states.NE,
        states.NV, states.NH, states.NJ, states.NM, states.NY, states.NC, states.ND,
        states.OH, states.OK, states.OR, states.PA, states.RI, states.SC, states.SD,
        states.TN, states.TX, states.UT, states.VT, states.VA, states.WA, states.WV,
        states.WI, states.WY
    ]

    # Define ACS variables
    acs_variables = [
        'NAME',
        'B01003_001E', 'B19013_001E', 'B19001_001E', 'B19001_002E', 'B19001_003E', 'B19001_004E', 'B19001_005E', 'B19001_006E', 'B19001_007E', 'B19001_008E', 'B19001_009E', 'B19001_010E', 'B19001_011E', 'B17001_001E', 'B17001_002E', 'B19083_001E', 'B25070_001E', 'B25070_007E', 'B25070_008E', 'B25070_009E', 'B25070_010E', 'B25064_001E', 'B25077_001E', 'B25034_001E', 'B25034_008E', 'B25034_009E', 'B25034_010E', 'B25034_011E', 'B25002_001E', 'B25002_003E', 'B25003_001E', 'B25003_002E', 'B25003_003E', 'B25004_003E', 'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E', 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E', 'B01001_020E', 'B01001_021E', 'B01001_044E', 'B01001_045E', 'B18101_004E', 'B18101_007E', 'B18101_010E', 'B18101_013E', 'B18101_016E', 'B18101_019E', 'B18101_023E', 'B18101_026E', 'B18101_029E', 'B18101_032E', 'B18101_035E', 'B18101_038E', 'B25118_001E', 'B25118_002E', 'B25118_003E', 'B25118_004E', 'B25118_005E', 'B25118_006E', 'B25118_007E', 'B25118_008E', 'B11001_001E', 'B11001_002E', 'B03002_001E', 'B03002_003E', 'B09010_001E'
    ]

    # List to store GeoDataFrames
    all_state_gdfs = []

    print(f"Starting download for {len(state_fips_list)} states/territories...")

    # Parallel download with ThreadPoolExecutor
    max_workers = 4  # Adjust based on API limits and system resources
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks for all states
        future_to_state = {
            executor.submit(download_state_data, state_fips, ACS5, ACS_VINTAGE, acs_variables): state_fips
            for state_fips in state_fips_list
        }
        print(f"Submitted {len(future_to_state)} tasks for processing...")

        # Collect results as they complete
        for future in as_completed(future_to_state):
            state_fips = future_to_state[future]
            try:
                gdf_tracts = future.result()
                if gdf_tracts is not None:
                    print(f"Successfully processed {state_fips} with {len(gdf_tracts)} tracts.")
                    all_state_gdfs.append(gdf_tracts)
                else:
                    print(f"No data returned for {state_fips}.")
            except Exception as e:
                print(f"Error processing {state_fips}: {e}")

    print("\n--- Download Complete ---")

    if all_state_gdfs:
        print(f"Concatenating data for {len(all_state_gdfs)} states/territories...")
        final_gdf = gpd.GeoDataFrame(pd.concat(all_state_gdfs, ignore_index=True), crs=all_state_gdfs[0].crs)

        print(f"Final GeoDataFrame shape: {final_gdf.shape}")
        print("Final GeoDataFrame Info:")
        final_gdf.info()
        print("\nSaving combined GeoDataFrame...")

        acs_variable_mapping_combined = {
            # --- Population ---
            'B01003_001E': 'Total_Population',

            # --- Income ---
            'B19013_001E': 'Median_Household_Income',
            'B19001_001E': 'Total_Households_Income_Breakdown',
            'B19001_002E': 'Households_Income_Less_10K',
            'B19001_003E': 'Households_Income_10K_15K',
            'B19001_004E': 'Households_Income_15K_20K',
            'B19001_005E': 'Households_Income_20K_25K',
            'B19001_006E': 'Households_Income_25K_30K',
            'B19001_007E': 'Households_Income_30K_35K',
            'B19001_008E': 'Households_Income_35K_40K',
            'B19001_009E': 'Households_Income_40K_45K',
            'B19001_010E': 'Households_Income_45K_50K',
            'B19001_011E': 'Households_Income_50K_60K',

            # --- Poverty ---
            'B17001_001E': 'Total_Population_Poverty_Status',
            'B17001_002E': 'Population_Below_Poverty',

            # --- Income Inequality ---
            'B19083_001E': 'Gini_Index_Income_Inequality',

            # --- Rent Burden ---
            'B25070_001E': 'Total_Renter_Occupied_Units_For_Rent_Burden',
            'B25070_007E': 'Rent_30_35_Percent_Income',
            'B25070_008E': 'Rent_35_40_Percent_Income',
            'B25070_009E': 'Rent_40_50_Percent_Income',
            'B25070_010E': 'Rent_50_Percent_Or_More_Income',

            # --- Median Rent ---
            'B25064_001E': 'Median_Gross_Rent',

            # --- Housing Value ---
            'B25077_001E': 'Median_Housing_Value',

            # --- Housing Stock (Age) ---
            'B25034_001E': 'Total_Housing_Units_Year_Built',
            'B25034_008E': 'Housing_Built_1960_1969',
            'B25034_009E': 'Housing_Built_1950_1959',
            'B25034_010E': 'Housing_Built_1940_1949',
            'B25034_011E': 'Housing_Built_1939_or_Earlier',

            # --- Housing Occupancy ---
            'B25002_001E': 'Total_Housing_Units',
            'B25002_003E': 'Vacant_Housing_Units',
            'B25003_001E': 'Total_Occupied_Housing_Units',
            'B25003_002E': 'Owner_Occupied_Housing_Units',
            'B25003_003E': 'Renter_Occupied_Housing_Units',
            'B25004_003E': 'Total_Renter_Occupied_Units_For_Rent',

            # --- Age & Sex: Elderly Population (65+) ---
            'B01001_022E': 'Male_65_66_Years',
            'B01001_023E': 'Male_67_69_Years',
            'B01001_024E': 'Male_70_74_Years',
            'B01001_025E': 'Male_75_Years_And_Over',
            'B01001_046E': 'Female_65_66_Years',
            'B01001_047E': 'Female_67_69_Years',
            'B01001_048E': 'Female_70_74_Years',
            'B01001_049E': 'Female_75_Years_And_Over',

            # --- Age & Sex: Additional Elderly Buckets ---
            'B01001_020E': 'Male_60_61_Years',
            'B01001_021E': 'Male_62_64_Years',
            'B01001_044E': 'Female_60_61_Years',
            'B01001_045E': 'Female_62_64_Years',

            # --- Disability by Age & Sex ---
            'B18101_004E': 'Male_Under_5_With_Disability',
            'B18101_007E': 'Male_5_17_With_Disability',
            'B18101_010E': 'Male_18_34_With_Disability',
            'B18101_013E': 'Male_35_64_With_Disability',
            'B18101_016E': 'Male_65_74_With_Disability',
            'B18101_019E': 'Male_75_Over_With_Disability',
            'B18101_023E': 'Female_Under_5_With_Disability',
            'B18101_026E': 'Female_5_17_With_Disability',
            'B18101_029E': 'Female_18_34_With_Disability',
            'B18101_032E': 'Female_35_64_With_Disability',
            'B18101_035E': 'Female_65_74_With_Disability',
            'B18101_038E': 'Female_75_Over_With_Disability',

            # --- Low-Income Renter Households ---
            'B25118_001E': 'Total_Renter_Occupied_HH_Income_Distribution',
            'B25118_002E': 'Renter_HH_Income_Less_5K',
            'B25118_003E': 'Renter_HH_Income_5K_10K',
            'B25118_004E': 'Renter_HH_Income_10K_15K',
            'B25118_005E': 'Renter_HH_Income_15K_20K',
            'B25118_006E': 'Renter_HH_Income_20K_25K',
            'B25118_007E': 'Renter_HH_Income_25K_35K',
            'B25118_008E': 'Renter_HH_Income_35K_50K',

            # --- Household Composition ---
            'B11001_001E': 'Total_Households',
            'B11001_002E': 'Family_Households',

            # --- Race & Ethnicity ---
            'B03002_001E': 'Total_Population_Race_Ethnicity',
            'B03002_003E': 'Not_Hispanic_White_Alone',

            # --- SSI ---
            'B09010_001E': 'Reciept_of_SSI'
        }

        final_gdf.rename(columns=acs_variable_mapping_combined, inplace=True)

        final_gdf.to_parquet(OUTPUT_FILENAME)
        print(f"Data saved successfully to {OUTPUT_FILENAME}")
    else:
        print("No data was successfully downloaded for any state. No file saved.")

    print("Script finished.")