import pandas as pd  
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from scipy.stats import ttest_ind
if __name__ == '__main__':
    # NHPD stands for the National Housing Preservation Database. It helps identify properties at risk of losing their affordability due to expiring subsidies, helping prevent displacement and preserve affordable housing.

    df = pd.read_excel('data/Active and Inconclusive Properties.xlsx')

    # Idea: Predicting and Preventing the loss of Affordable Housing: Risk Assessment of Subsidized Properties

    # Objective: Develop a risk assessment framework to identify subsidized housing properties at greatest risk of losing their affordability status due to expiring subsidies. 

    df['EarliestEndDate'] = pd.to_datetime(df['EarliestEndDate'], errors='coerce')
    df['LatestEndDate'] = pd.to_datetime(df['LatestEndDate'], errors='coerce')

    df_v1  = df[df['PropertyStatus'] == 'Active']
    current_date = datetime.today()

    # Time windows for expiration analysis
    five_years = current_date + pd.DateOffset(years=5)
    ten_years = current_date + pd.DateOffset(years=10)
    
    expiring_5yr = df_v1[(df_v1['LatestEndDate'] >= current_date) & (df_v1['LatestEndDate'] <= five_years)].copy()
    expiring_10yr = df_v1[(df_v1['LatestEndDate'] >= current_date) & (df_v1['LatestEndDate'] <= ten_years)].copy()

    # Calculate total units at risk
    units_5yr = expiring_5yr['TotalUnits'].sum()
    units_10yr = expiring_10yr['TotalUnits'].sum()

    # Summarize by state
    summary_5yr_by_state = expiring_5yr.groupby('State')['TotalUnits'].sum().reset_index()
    summary_10yr_by_state = expiring_10yr.groupby('State')['TotalUnits'].sum().reset_index()

    # Summarize by city (optional)
    summary_5yr_by_city = expiring_5yr.groupby(['State', 'City'])['TotalUnits'].sum().reset_index()
    summary_10yr_by_city = expiring_10yr.groupby(['State', 'City'])['TotalUnits'].sum().reset_index()

    print("Subsidy Expiration Snapshot (as of March 03, 2025):")
    print(f"Total units with subsidies expiring within 5 years (by {five_years.date()}): {units_5yr}")
    print(f"Total units with subsidies expiring within 10 years (by {ten_years.date()}): {units_10yr}")
    print("\nUnits at risk within 5 years by State:")
    print(summary_5yr_by_state)
    print("\nUnits at risk within 10 years by State:")
    print(summary_10yr_by_state)
    print("\nUnits at risk within 5 years by City:")
    print(summary_5yr_by_city)
    print("\nUnits at risk within 10 years by City:")
    print(summary_10yr_by_city)

    plt.figure(figsize=(10, 6))
    plt.bar(summary_10yr_by_state['State'], summary_10yr_by_state['TotalUnits'], color='skyblue')
    plt.xlabel('State')
    plt.ylabel('Total Units at Risk')
    plt.title('Subsidized Units with Expiring Subsidies by State (2025-2035)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



    summary_10yr_by_state['TotalUnits_normalized'] = summary_10yr_by_state['TotalUnits'] / summary_10yr_by_state['TotalUnits'].max()

    # In Relative terms, New York State has the highest number of units at risk, followed by California and Texas in the next 10 years. This provides an immediate view of the affordable housing stock at risk, helping HUD plan renewals or alternative funding

    # Folium Map: Properties with 5-year (blue) and 10-year (orange) expiration windows
    
    # Initialize map centered on the US
    import folium
    from folium.plugins import MarkerCluster
    from folium import Map, Marker, LayerControl, Element

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    five_yr_cluster = MarkerCluster(name='5-Year Expirations').add_to(m)
    ten_yr_cluster = MarkerCluster(name='10-Year Expirations').add_to(m)
    # Add title
    title_html = '''
        <h3 align="center" style="font-size:20px"><b>Subsidized Housing Expiration Risk (5 & 10 Year Outlook)</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 5-year markers (blue)
    for _, row in expiring_5yr.iterrows():
        popup = f"""
        <div style='width: 250px;'>
        <b>Property Name:</b> {row['PropertyName']}<br>
        <b>Number of Units:</b> {row['TotalUnits']}<br>
        <b>Expiration Date:</b> {row['LatestEndDate'].date()}<br>
        </div>
        """
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=popup,
            icon=folium.Icon(color='blue')
        ).add_to(five_yr_cluster)

    # 10-year markers (orange)
    for _, row in expiring_10yr.iterrows():
        if row['LatestEndDate'] <= five_years:
            continue
        popup = f"""
        <div style='width: 250px;'>
        <b>Property Name:</b> {row['PropertyName']}<br>
        <b>Number of Units:</b> {row['TotalUnits']}<br>
        <b>Expiration Date:</b> {row['LatestEndDate'].date()}<br>
        </div>
        """
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=popup,
            icon=folium.Icon(color='orange')
        ).add_to(ten_yr_cluster)

    # Layer control
    folium.LayerControl().add_to(m)



    # Save map
    m.save('subsidy_expiration_map_with_title_and_footnote.html')

    # This map provides an immediate view of the affordable housing stock at risk of expiration, helping HUD plan renewals or alternative funding. 

    # Property Condition Analysis
    
    # I noticed that there are 3 score features (Real estate assessment center scores) -  ReacScore1,ReacScore2, ReacScore3. This is missing for a lot of properties. For the properties that has atleast 1 non-missing values, I will compute an average of this score.

    # This will help inform poorly maintained properties for example (scores below 50) may signal risk of subsidy loss or converstion to market rate housing informing HUD's inspection priorities. 

    len(df_v1) # in total 77448 have active property status

    df_v3 = df_v1[['PropertyName','Latitude','Longitude','ReacScore1','ReacScore2','ReacScore3','OwnerType']]
    df_v3.isna().sum() # there are 49484 properties missing these scores. I will only consider properties with atleast 1 non-missing value

    df_v4 = df_v3.dropna(subset=['ReacScore1', 'ReacScore2', 'ReacScore3'], how='all')



 

    # Define standardization dictionary for OwnerType
    standard_owner_types = {
        'limited dividend': 'Limited Dividend',
        'non profit': 'Non-Profit',
        'non-profit': 'Non-Profit',
        'profit motivated': 'Profit Motivated',
        'for profit': 'For Profit',
        'limited profit': 'Limited Profit',
        'multiple': 'Multiple',
        'public entity': 'Public Entity'
    }

    # Define function to standardize OwnerType
    def standardize_owner_type(owner_type):
        if pd.isna(owner_type):
            return np.nan
        lower_type = owner_type.lower().strip()
        return standard_owner_types.get(lower_type, owner_type.title())

    # Create df_v3 from df_v1
    df_v3 = df_v1[['PropertyName', 'Latitude', 'Longitude', 'ReacScore1', 'ReacScore2', 'ReacScore3', 'OwnerType']]

    # Apply standardization to OwnerType
    df_v3['OwnerType'] = df_v3['OwnerType'].apply(standardize_owner_type)

    # Check missing values (optional, for verification)
    print("Missing values in df_v3:")
    print(df_v3.isna().sum())

    # Filter to keep properties with at least one non-missing REAC score
    df_v4 = df_v3.dropna(subset=['ReacScore1', 'ReacScore2', 'ReacScore3'], how='all')

    # Function to parse REAC scores
    def parse_reac_score(score):
        if pd.isna(score) or not isinstance(score, str):
            return np.nan
        numeric_part = ''.join(filter(str.isdigit, score))
        return float(numeric_part) if numeric_part else np.nan

    # Apply parsing to REAC score columns
    df_v4['ReacScore1_num'] = df_v4['ReacScore1'].apply(parse_reac_score)
    df_v4['ReacScore2_num'] = df_v4['ReacScore2'].apply(parse_reac_score)
    df_v4['ReacScore3_num'] = df_v4['ReacScore3'].apply(parse_reac_score)

    # Compute average REAC score, ignoring NaN values
    df_v4['AvgReacScore'] = df_v4[['ReacScore1_num', 'ReacScore2_num', 'ReacScore3_num']].mean(axis=1, skipna=True)

    # Flag at-risk properties with average score below 60
    df_v4['AtRisk'] = df_v4['AvgReacScore'] < 60

    # Summarize by OwnerType
    summary_by_owner = df_v4.groupby('OwnerType').agg({
        'AvgReacScore': ['mean', 'count'],
        'AtRisk': 'sum'
    }).reset_index()

    # Rename columns for clarity
    summary_by_owner.columns = ['OwnerType', 'AvgScore_Mean', 'PropertyCount', 'AtRiskCount']
    print("\nSummary by OwnerType after standardization:")
    print(summary_by_owner)

    # Visualizations

    # Boxplot of REAC scores by OwnerType
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OwnerType', y='AvgReacScore', data=df_v4)
    plt.axhline(60, color='red', linestyle='--', label='Failing Threshold (60)')
    plt.title('Property Condition by Owner Type')
    plt.xlabel('Owner Type')
    plt.ylabel('Average REAC Score')
    plt.xticks(rotation=45)  # Rotate labels for readability
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Shows the spread of scores and highlights if certain owner types consistently fall below 60.
    
    # Barplot of at-risk properties by OwnerType
    plt.figure(figsize=(10, 6))
    sns.barplot(x='OwnerType', y='AtRiskCount', data=summary_by_owner, palette='Reds')
    plt.title('Number of At-Risk Properties by Owner Type')
    plt.xlabel('Owner Type')
    plt.ylabel('Number of Properties with Avg REAC < 60')
    plt.xticks(rotation=45)  # Rotate labels for readability
    plt.tight_layout()
    plt.show()
    # Quantifies how many properties are at risk per owner type, emphasizing disparities.


    # Initialize Folium map centered on the U.S.
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    at_risk_cluster = MarkerCluster(name='At-Risk Properties (Avg REAC < 60)').add_to(m)
    safe_cluster = MarkerCluster(name='Safe Properties (Avg REAC >= 60)').add_to(m)

    # Add markers to the map
    for _, row in df_v4.iterrows():
        if pd.isna(row['AvgReacScore']):
            continue
        color = 'red' if row['AtRisk'] else 'green'
        cluster = at_risk_cluster if row['AtRisk'] else safe_cluster
        popup = f"""
        <div style='width: 250px;'>
        <b>Property Name:</b> {row['PropertyName']}<br>
        <b>Avg REAC Score:</b> {row['AvgReacScore']:.1f}<br>
        </div>
        """
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=popup,
            icon=folium.Icon(color=color)
        ).add_to(cluster)


    folium.LayerControl().add_to(m)
    m.save('property_condition_map.html')
    print("Map saved as 'property_condition_map.html'.")
    
    # Red markers for at-risk properties and green for safe ones, clustered for readability. This could guide HUD to focus inspections in high-risk areas.

    # T-test between Non-Profit and Profit Motivated
    nonprofit_scores = df_v4[df_v4['OwnerType'] == 'Non-Profit']['AvgReacScore'].dropna()
    profit_scores = df_v4[df_v4['OwnerType'] == 'Profit Motivated']['AvgReacScore'].dropna()
    t_stat, p_val = ttest_ind(nonprofit_scores, profit_scores)
    print(f"T-test p-value (Non-Profit vs. Profit Motivated): {p_val:.4f}")

    # The p-value is less than 0.05, indicating a statistically significant difference between the average REAC scores of nonprofit and profit-motivated properties.

    # housing_gap measures the absolute shortage in units against a 5% benchmark (e.g., 5% of elderly population should have access), providing a concrete, interpretable number
    # Correlations and regression now assess how housing gaps relate to economic need

    # Load and standardize housing data
    df_v2 = df_v1.copy()
    df_v1 = df_v1[['NHPDPropertyID', 'PropertyName', 'CensusTract', 'State', 'TotalUnits', 
                'TargetTenantType', 'FairMarketRent_2BR', 'PercentofELIHouseholds']].copy()

    # Standardize TargetTenantType
    standard_mapping = {
        'Disabled': 'Disabled',
        'Elderly': 'Elderly',
        'Senior': 'Elderly',
        'Family': 'Family',
        'FAMILY': 'Family',
        'Low Income': 'Low Income',
        'Elderly or disabled': 'Elderly or Disabled',
        'Eldery or Disabled': 'Elderly or Disabled',
        'Mixed': 'Mixed',
        'MIXED': 'Mixed',
        'Homeless': 'Homeless',
        'Homeless Veterans': 'Homeless Veterans',
        'Affordable': 'Low Income',
        'Veterans': 'Veterans',
        'Congregate': 'Congregate',
        'Farmworkers': 'Farmworkers',
        'Mixed Income': 'Mixed Income',
        'Special needs': 'Special Needs',
        'Group Home': 'Group Home',
        'Supportive Housing': 'Supportive Housing',
        'OTHER': 'Other'
    }

    def standardize_tenant_type(tenant_type):
        return standard_mapping.get(tenant_type, 'Other') if pd.notna(tenant_type) else 'Other'

    df_v1['TargetTenantType'] = df_v1['TargetTenantType'].apply(standardize_tenant_type)
    print("Standardized Tenant Types:", df_v1['TargetTenantType'].unique())

    # Load ACS data
    census_tract_df = pd.read_csv("data/tract_census_data_all_states.csv")
    census_tract_df['CensusTract'] = census_tract_df['CensusTract'].astype(str)
    df_v1['CensusTract'] = df_v1['CensusTract'].apply(lambda x: f'{int(x):011d}' if pd.notna(x) else '')
    df_v2['CensusTract'] = df_v2['CensusTract'].apply(lambda x: f'{int(x):011d}' if pd.notna(x) else '')
    # Merge datasets
    df_merged = pd.merge(df_v2, census_tract_df, on='CensusTract', how='left')
    df_merged.to_csv("data/df_merged.csv",index = False)
    df_merged = df_merged[df_merged['CensusTract'] != ''].copy()

    # Define key tenant types and calculate housing gaps
    key_tenant_types = ['Elderly', 'Disabled', 'Family', 'Low Income']

    def calculate_housing_gap(row):
        # Benchmark: 5% of target population should have subsidized units
        benchmark = 0.05
        if row['TargetTenantType'] == 'Elderly':
            target_pop = row['Population_65_and_over']
        elif row['TargetTenantType'] == 'Disabled':
            target_pop = row['Population_with_disability']
        elif row['TargetTenantType'] == 'Family':
            target_pop = row['B01003_001E'] * 0.5  # Assume 50% are in family households
        elif row['TargetTenantType'] == 'Low Income':
            target_pop = row['B01003_001E'] * (row['pct_cost_burdened_renters'] / 100)
        else:
            return np.nan
        needed_units = target_pop * benchmark if target_pop > 0 else 0
        gap = needed_units - row['TotalUnits']
        return gap if gap > 0 else 0  # Positive gap = shortage

    df_merged['housing_gap'] = df_merged.apply(calculate_housing_gap, axis=1)
    df_analysis = df_merged[df_merged['TargetTenantType'].isin(key_tenant_types)].copy()

    # Summary Statistics for Housing Gaps
    for tenant_type in key_tenant_types:
        subset = df_analysis[df_analysis['TargetTenantType'] == tenant_type]
        print(f"\nHousing Gap Statistics for {tenant_type} (Units Needed):")
        print(subset['housing_gap'].describe())

    # State-Level Housing Gap Analysis
    print("\nTotal Housing Gap by State and Tenant Type (Units Needed):")
    state_gaps = df_analysis.groupby(['State', 'TargetTenantType'])['housing_gap'].sum().unstack().fillna(0)
    print(state_gaps)

    # Economic Alignment Analysis
    econ_vars = ['pct_cost_burdened_renters', 'FairMarketRent_2BR', 'PercentofELIHouseholds']
    df_econ = df_analysis.dropna(subset=['housing_gap'] + econ_vars)
    for var in econ_vars:
        corr = df_econ['housing_gap'].corr(df_econ[var])
        print(f"Correlation with {var}: {corr:.3f}")

  
    # Visualizations
    # 1. Boxplot of Housing Gaps by Tenant Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='TargetTenantType', y='housing_gap', data=df_analysis)
    plt.title('Subsidized Housing Shortages by Tenant Type')
    plt.xlabel('Target Tenant Type')
    plt.ylabel('Housing Gap (Units Needed)')
    plt.xticks(rotation=45)
    plt.ylim(0, 500)  # Adjust based on data range
    plt.show()

    # 2. Choropleth Map for Elderly Housing Gap by State
    state_geo = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json'
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # Define color schemes for each tenant type
    colors = {
        'Elderly': 'YlOrRd',
        'Disabled': 'Blues',
        'Family': 'Greens',
        'Low Income': 'Purples'
    }

    # Add a choropleth layer for each tenant type
    for tenant_type in key_tenant_types:
        folium.Choropleth(
            geo_data=state_geo,
            name=f'{tenant_type} Housing Gap',
            data=state_gaps[tenant_type],
            columns=[state_gaps.index, tenant_type],
            key_on='feature.id',
            fill_color=colors[tenant_type],
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f'{tenant_type} Housing Gap (Units Needed)',
            show=(tenant_type == 'Elderly')  # Only show Elderly by default
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save('housing_gap_choropleth_all_types.html')
    print("Choropleth map with all tenant types saved to 'housing_gap_choropleth_all_types.html'")

    # 3. Scatter Plot: Housing Gap vs. Cost Burden (unchanged)
    plt.figure(figsize=(10, 6))
    for tenant_type in key_tenant_types:
        subset = df_econ[df_econ['TargetTenantType'] == tenant_type]
        plt.scatter(subset['pct_cost_burdened_renters'], subset['housing_gap'], 
                    label=tenant_type, alpha=0.5)
    plt.title('Housing Gap vs. Percentage of Cost-Burdened Renters')
    plt.xlabel('Percentage of Cost-Burdened Renters')
    plt.ylabel('Housing Gap (Units Needed)')
    plt.legend(title='Tenant Type')
    plt.show()

    # Save results
    df_analysis.to_csv("data/merged_housing_gap_analysis.csv", index=False)
    print("Analysis saved to 'data/merged_housing_gap_analysis.csv'")