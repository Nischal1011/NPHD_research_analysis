import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/df_merged.csv", low_memory=False)

# Composite End Date
end_date_cols = [
    'S8_1_EndDate', 'S8_2_EndDate', 'S202_1_EndDate', 'S202_2_EndDate', 
    'S236_1_EndDate', 'S236_2_EndDate', 'FHA_1_EndDate', 'FHA_2_EndDate',
    'LIHTC_1_EndDate', 'LIHTC_2_EndDate', 'RHS515_1_EndDate', 'RHS515_2_EndDate',
    'RHS538_1_EndDate', 'RHS538_2_EndDate', 'HOME_1_EndDate', 'HOME_2_EndDate',
    'PH_1_EndDate', 'PH_2_EndDate', 'State_1_EndDate', 'State_2_EndDate',
    'NHTF_1_EndDate', 'NHTF_2_EndDate'
]
status_cols = [
    'S8_1_Status', 'S8_2_Status', 'S202_1_Status', 'S202_2_Status',
    'S236_1_Status', 'S236_2_Status', 'FHA_1_Status', 'FHA_2_Status',
    'LIHTC_1_Status', 'LIHTC_2_Status', 'RHS515_1_Status', 'RHS515_2_Status',
    'RHS538_1_Status', 'RHS538_2_Status', 'HOME_1_Status', 'HOME_2_Status',
    'PH_1_Status', 'PH_2_Status', 'State_1_Status', 'State_2_Status',
    'NHTF_1_Status', 'NHTF_2_Status'
]

for col in end_date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

def get_composite_end_date(row):
    active_end_dates = [row[end_col] for end_col, status_col in zip(end_date_cols, status_cols)
                        if pd.notna(row[end_col]) and row.get(status_col, '') == 'Active']
    return max(active_end_dates) if active_end_dates else pd.NaT

df['CompositeEndDate'] = df.apply(get_composite_end_date, axis=1)
df['DaysToExpiration_Composite'] = (df['CompositeEndDate'] - pd.Timestamp.today()).dt.days
df['Expiration_5yr'] = df['DaysToExpiration_Composite'].between(0, 5 * 365).astype(int)  # Binary

# Affordability Risk
income_threshold = df['B19013_001E'].quantile(0.25)
df['High_Risk_Affordability'] = ((df['pct_cost_burdened_renters'] > 50) & 
                                 (df['B19013_001E'] < income_threshold)).astype(int)

# Condition Proxy
df['ConstructionAge'] = (pd.Timestamp.today() - pd.to_datetime(df['EarliestConstructionDate'], errors='coerce')).dt.days
df['Condition_Proxy'] = (df['ConstructionAge'] > 50 * 365).astype(int)

# Combined Risk
df['High_Risk_Combined'] = ((df['High_Risk_Affordability'] == 1) | 
                            (df['Expiration_5yr'] == 1) | 
                            (df['Condition_Proxy'] == 1)).astype(int)
print("Final Combined Risk Counts:")
print(df['High_Risk_Combined'].value_counts())

# Individual Breakdown
print("\nIndividual Risk Counts:")
print(f"Affordability Risk: {df['High_Risk_Affordability'].sum()}")
print(f"Expiration_5yr Risk: {df['Expiration_5yr'].sum()}")
print(f"Condition Proxy Risk: {df['Condition_Proxy'].sum()}")

# Overlap Breakdown
print("\nOverlap Breakdown:")
print(f"Affordability Only: {((df['High_Risk_Affordability'] == 1) & (df['Expiration_5yr'] == 0) & (df['Condition_Proxy'] == 0)).sum()}")
print(f"Expiration Only: {((df['High_Risk_Affordability'] == 0) & (df['Expiration_5yr'] == 1) & (df['Condition_Proxy'] == 0)).sum()}")
print(f"Condition Only: {((df['High_Risk_Affordability'] == 0) & (df['Expiration_5yr'] == 0) & (df['Condition_Proxy'] == 1)).sum()}")
print(f"Affordability + Expiration: {((df['High_Risk_Affordability'] == 1) & (df['Expiration_5yr'] == 1) & (df['Condition_Proxy'] == 0)).sum()}")
print(f"Affordability + Condition: {((df['High_Risk_Affordability'] == 1) & (df['Expiration_5yr'] == 0) & (df['Condition_Proxy'] == 1)).sum()}")
print(f"Expiration + Condition: {((df['High_Risk_Affordability'] == 0) & (df['Expiration_5yr'] == 1) & (df['Condition_Proxy'] == 1)).sum()}")
print(f"All Three: {((df['High_Risk_Affordability'] == 1) & (df['Expiration_5yr'] == 1) & (df['Condition_Proxy'] == 1)).sum()}")

df.rename(columns={
    'B18101_029E': 'Census_Age_25_44',
    'B25070_009E': 'Census_Median_Home_Value',
    'B15003_022E': 'Census_Educational_Attainment_College_Degree',
    'B25070_010E': 'Census_Avg_Home_Value',
    'B01001_044E': 'Census_Population_25_44',
    'B18101_038E': 'Census_Age_45_64',
    'B18101_007E': 'Census_Age_18_24',
    'B18101_035E': 'Census_Age_65_74',
    'B18101_026E': 'Census_Age_75_and_older',
    'B25070_008E': 'Census_Median_Home_Value_Owner_Occupied',
    'B19013_001E': 'Census_Median_Income',
    'B07003_004E': 'Census_Migration_In',
    'B25070_007E': 'Census_Median_Home_Value_Empty',
    'B11001_001E': 'Census_Household_Head_Type',
    'B01001_022E': 'Census_Population_45_64',
    'B18101_016E': 'Census_Age_25_34',
    'B25070_001E': 'Census_Median_Home_Value_Overall',
    'B19083_001E': 'Census_Median_Income_by_Household',
    'B02001_003E': 'Census_White_Population',
    'B07001_033E': 'Census_Population_Moved_Within_Last_Year',
    'B01001_045E': 'Census_Population_45_64_Men',
    'B25003_003E': 'Census_Median_Apartment_Rent',
    'B23025_005E': 'Census_Labor_Force_Participation',
    'B25003_002E': 'Census_Median_Household_Rent',
    'B18101_004E': 'Census_Age_35_44',
    'B25034_004E': 'Census_Year_Built_1950_1979',
    'B18101_023E': 'Census_Age_55_64',
    'B01001_021E': 'Census_Population_35_44',
    'B02001_005E': 'Census_Black_Population',
    'B07001_017E': 'Census_Migration_Out',
    'B01001_023E': 'Census_Population_50_59',
    'B02001_002E': 'Census_Asian_Population',
    'B18101_013E': 'Census_Age_65_74_Alt',
    'B11001_002E': 'Census_Single_Headed_Households',
    'B25034_009E': 'Census_Year_Built_1980_1999',
    'B25034_008E': 'Census_Year_Built_2000_and_Above',
    'B25064_001E': 'Census_Avg_Home_Size',
    'B11001_007E': 'Census_Married_Couple_Family',
    'B25034_006E': 'Census_Year_Built_1970_1979',
    'B25034_010E': 'Census_Year_Built_2010_Above',
    'B01001_001E': 'Census_Population_Total',
    'B01003_001E': 'Census_Total_Population',
    'B07003_010E': 'Census_Year_Stayed_Population',
    'B25077_001E': 'Census_Median_Contract_Rent',
    'B03001_003E': 'Census_Hispanic_Population',
    'B01001_020E': 'Census_Population_60_69',
    'B25004_001E': 'Census_Median_House_Rent',
    'B01001_048E': 'Census_Population_75_84',
    'B01001_049E': 'Census_Population_85_and_older',
    'NAME': 'Census_Region_Name',
    'state_fips': 'Census_State_FIPS',
    'county_fips': 'Census_County_FIPS',
    'tract_fips': 'Census_Tract_FIPS',
    'B01001_025E': 'Census_Population_30_39',
    'B01001_046E': 'Census_Population_60_69_Men',
    'B07001_049E': 'Census_Immigrants_Moved_Within_Last_Year',
    'B18101_010E': 'Census_Age_15_24',
    'B01001_024E': 'Census_Population_40_49',
    'B18101_032E': 'Census_Age_50_59',
    'B25034_007E': 'Census_Year_Built_1960_1969',
    'B25034_002E': 'Census_Year_Built_1940_1949',
    'B25034_005E': 'Census_Year_Built_1950_1959',
    'B25034_003E': 'Census_Year_Built_1960_1969',
    'B01001_006E': 'Census_Population_20_29',
    'B01001_047E': 'Census_Population_50_59_Men',
    'B18101_019E': 'Census_Age_55_64_Alt',
    'Population_65_and_over': 'Census_Population_65_And_Over',
    'Population_with_disability': 'Census_Disability_Population',
    'pct_cost_burdened_renters': 'Census_Percentage_Cost_Burdened_Renters',
    'CompositeEndDate': 'Property_Composite_End_Date',
    'DaysToExpiration_Composite': 'Property_Days_To_Expiration',
    'Expiration_5yr': 'Property_5_Year_Expiration',
    'High_Risk_Affordability': 'Risk_Affordability',
    'ConstructionAge': 'Property_Construction_Age',
    'Condition_Proxy': 'Property_Condition_Proxy',
    'High_Risk_Combined': 'Risk_Combined'
}, inplace=True)

df.to_csv("data/df_with_target.csv", index = False)
