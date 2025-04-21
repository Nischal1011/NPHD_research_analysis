import pandas as pd  
import os
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from scipy.stats import ttest_ind
import branca.colormap as cm
from folium.plugins import FloatImage
import geopandas as gpd
import re
from shapely import wkb
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from branca.colormap import StepColormap


def plot_rent_ratio_boxplot(data, title="Boxplot of Rent Ratio", ylabel="Rent Ratio", figsize=(10, 8), save_path="data/charts/rent_ratio_boxplot.png"):
    """
    Create and save a boxplot for RentRatio data with enhanced visuals for research use.

    Parameters:
    - data: Pandas Series or array-like, containing RentRatio values.
    - title: String, title of the plot.
    - ylabel: String, label for y-axis.
    - figsize: Tuple, figure size as (width, height).
    - save_path: String, where to save the plot image.

    Returns:
    - Saves the boxplot to the specified path.
    """
    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create figure
    plt.figure(figsize=figsize)

    # Plot
    sns.boxplot(y=data, color="skyblue", width=0.4)

    # Mean line
    mean_val = np.mean(data)
    plt.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")

    # Customizations
    plt.title(title, fontsize=18, pad=20)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save with enough padding
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_rent_ratio_choropleth(gdf, rent_ratio_col='RentRatio', 
                                 output_file='data/rent_ratio_choropleth.html'):
        """
        Create an interactive Folium choropleth map of RentRatio from a GeoDataFrame.
        
        Parameters:
        - gdf: GeoDataFrame, contains RentRatio and geometry columns.
        - rent_ratio_col: String, name of the RentRatio column (default: 'RentRatio').
        - title: String, title of the map (default: "Market Rent Pressure").
        - output_file: String, path to save the HTML map (default: 'data/rent_ratio_choropleth.html').
        
        Returns:
        - Folium Map object.
        """
        # Validate input
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("Input 'gdf' must be a GeoDataFrame.")
        if rent_ratio_col not in gdf.columns:
            raise ValueError(f"Column '{rent_ratio_col}' not found in GeoDataFrame.")
        
        # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
        if gdf.crs is None:
            print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
            gdf = gdf.set_crs(epsg=4326)
        
        # Reproject to a projected CRS (Albers Equal Area, EPSG:5070) for centroid calculation
        gdf_projected = gdf.to_crs(epsg=5070)
        
        # Calculate the centroid in the projected CRS
        centroid_projected = gdf_projected.geometry.centroid
        
        # Reproject the centroid back to EPSG:4326 for Folium
        centroid = centroid_projected.to_crs(epsg=4326)
        map_center = [centroid.y.mean(), centroid.x.mean()]
        
        # Reproject the GeoDataFrame to EPSG:4326 for Folium
        gdf = gdf.to_crs(epsg=4326)
        
        # Initialize the Folium map
        m = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")
        
        # Define classification bins and labels
        bins = [0, 0.2, 0.4, 0.6, 0.8, float('inf')]
        labels = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '>0.8']
        
        # Create a new column for binned categories
        gdf['RentRatioBin'] = pd.cut(
            gdf[rent_ratio_col],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Define color scheme (light green to dark green)
        color_map = {
            '0–0.2': '#E6F3E6',  # Light green
            '0.2–0.4': '#A8D5A8',  # Medium-light green
            '0.4–0.6': '#6BB76B',  # Medium green
            '0.6–0.8': '#3D993D',  # Medium-dark green
            '>0.8': '#006400'     # Dark green
        }
        
        # Create a stepped colormap for legend
        colormap = StepColormap(
            colors=list(color_map.values()),
            index=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Adjust index for legend
            vmin=0,
            vmax=1.0,
            caption='Rent Ratio'
        )
        
        # Format RentRatio for tooltips
        gdf['RentRatioFormatted'] = gdf[rent_ratio_col].apply(lambda x: f"{x:.3f}")
        
        # Define style function
        def style_function(feature):
            bin_label = feature['properties']['RentRatioBin']
            return {
                'fillColor': color_map.get(bin_label, '#ffffff'),  # Default to white if bin is missing
                'color': 'black',
                'weight': 0.1,
                'fillOpacity': 0.7
            }
        
        # Prepare tooltip fields (CensusTract first, then RentRatio)
        tooltip_fields = []
        tooltip_aliases = []
        if 'CensusTract' in gdf.columns:
            tooltip_fields.append('CensusTract')
            tooltip_aliases.append('Census Tract')
        tooltip_fields.append('RentRatioFormatted')
        tooltip_aliases.append('Rent Ratio')
        
        # Add GeoJson layer
        folium.GeoJson(
            gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                labels=True
            ),
            name='Rent Ratio'
        ).add_to(m)
        
        # Add colormap to map
        colormap.add_to(m)
        
        # Add footnote at the bottom using custom HTML
        footnote = """
        <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                    background-color: white; padding: 5px; border: 1px solid black; z-index: 1000;">
            <p style="margin: 0; font-size: 12px;">
                From an affordability perspective: Lower is better (indicates less rent pressure).
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(footnote))
        
        # Add title as HTML
        
        
        # Add LayerControl
        folium.LayerControl().add_to(m)
        
        # Save the map
        m.save(output_file)
        
        # Return the map
        return m

def plot_subsidized_housing_boxplot(data_series, 
        title="Subsidized Housing Coverage Ratio\n(Units per 1 Low-Income Renter Household)", 
        ylabel="Coverage Ratio", xlabel="Subsidized Housing", 
        save_path="data/charts/subsidized_housing_boxplot.png"):
        """
        Generate a vertical box plot to visualize the distribution of subsidized housing coverage ratios.

        The y-axis is capped at 0.6 to focus on the primary data range. Outliers are shown as small dots,
        and the plot includes annotations for the first quartile (Q1), third quartile (Q3), and interquartile range (IQR).
        
        Parameters:
        - data_series (pd.Series or array-like): Input data to visualize.
        - title (str): Title of the plot.
        - ylabel (str): Label for the y-axis.
        - xlabel (str): Label for the x-axis.
        - save_path (str): File path to save the plot.

        Returns:
        - float or None: The IQR value, or None if the input is invalid.
        """
        try:
            # Ensure input is a clean Series
            data_series = pd.Series(data_series).dropna()
            if len(data_series) == 0:
                raise ValueError("Data series is empty after dropping NaNs.")

            # Compute quartiles and IQR
            Q1 = data_series.quantile(0.25)
            Q3 = data_series.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR

            # Identify statistical outliers and visually clipped values
            outliers = data_series[data_series > upper_bound]
            clipped_points = data_series[data_series > 0.6]
            print(f"Outliers (> Q3 + 1.5×IQR): {len(outliers)} ({len(outliers)/len(data_series)*100:.2f}%)")
            print(f"Clipped values (> 0.6): {len(clipped_points)} ({len(clipped_points)/len(data_series)*100:.2f}%)")

            # Create the box plot
            plt.figure(figsize=(8, 6))
            plt.boxplot(data_series, vert=True, patch_artist=True, showfliers=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red'),
                flierprops=dict(marker='.', markersize=1, 
                                markerfacecolor='black', markeredgecolor='black', alpha=0.2)
            )

            # Set axis labels and range
            plt.title(title, fontsize=12)
            plt.ylabel(ylabel, fontsize=10)
            plt.xticks([1], [xlabel], fontsize=10)
            plt.ylim(0, 0.6)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Annotate Q1, Q3, and IQR (limit Q3 label to avoid visual cutoff)
            plt.text(1.1, min(Q3, 0.58), f'Q3: {Q3:.3f}', fontsize=10, va='center', color='blue')
            plt.text(1.1, (Q1 + min(Q3, 0.58)) / 2, f'IQR: {IQR:.3f}', fontsize=10, va='center', color='red')
            plt.text(1.1, Q1, f'Q1: {Q1:.3f}', fontsize=10, va='center', color='blue')

            # Save and display the plot
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {os.path.abspath(save_path)}")
            plt.show()

            return IQR

        except Exception as e:
            print(f"Error creating box plot: {e}")
            return None
def create_housing_coverage_choropleth(gdf):
        
        # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
        if gdf.crs is None:
            print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
            gdf = gdf.set_crs(epsg=4326)
        
        # Reproject to a projected CRS (Albers Equal Area, EPSG:5070) for centroid calculation
        gdf_projected = gdf.to_crs(epsg=5070)
        
        # Calculate the centroid in the projected CRS
        centroid_projected = gdf_projected.geometry.centroid
        
        # Reproject the centroid back to EPSG:4326 for Folium
        centroid = centroid_projected.to_crs(epsg=4326)
        map_center = [centroid.y.mean(), centroid.x.mean()]
        
        # Reproject the GeoDataFrame to EPSG:4326 for Folium
        gdf = gdf.to_crs(epsg=4326)
        
        # Initialize the Folium map
        m = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")
        
        # Define classification bins and labels
        bins = [0, 0.2, 0.4, 0.6, 0.8, float('inf')]
        labels = ['[0–0.2]', '(0.2–0.4]', '(0.4–0.6]', '(0.6–0.8]', '>0.8']
        
        # Create a new column for binned categories
        gdf['CoverageRatioBin'] = pd.cut(
            gdf['Subsidized_Housing_Coverage_Ratio'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Define color scheme (dark red to light orange)
        color_map = {
            '[0–0.2]': '#8B0000',  # Dark red
            '(0.2–0.4]': '#CD5C5C',  # Red-orange
            '(0.4–0.6]': '#FF4500',  # Medium orange
            '(0.6–0.8]': '#FFA500',  # Light orange
            '>0.8': '#FFE4B5'  # Lightest orange
        }
        
        # Create a stepped colormap for legend
        colormap = cm.StepColormap(
            colors=list(color_map.values()),
            index=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Adjust index for legend
            vmin=0,
            vmax=1.0,
            caption='Subsidized Housing Coverage Ratio'
        )
        
        # Format Subsidized_Housing_Coverage_Ratio for tooltips
        gdf['CoverageRatioFormatted'] = gdf['Subsidized_Housing_Coverage_Ratio'].apply(lambda x: f"{x:.3f}")
        
        # Define style function
        def style_function(feature):
            bin_label = feature['properties']['CoverageRatioBin']
            return {
                'fillColor': color_map.get(bin_label, '#ffffff'),  # Default to white if bin is missing
                'color': 'black',
                'weight': 0.1,
                'fillOpacity': 0.7
            }
        
        # Add GeoJson layer
        folium.GeoJson(
            gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'Total_LowIncome_Renter_HH', 'TotalUnits', 'CoverageRatioFormatted'],
                aliases=['Census Tract', 'Low-Income Renter Households', 'Subsidized Units', 'Coverage Ratio'],
                localize=True,
                labels=True
            ),
            name='Subsidized Housing Coverage'
        ).add_to(m)
        
        # Add colormap to map
        colormap.add_to(m)
        
        # Add footnote at the bottom using custom HTML
        footnote = """
        <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                    background-color: white; padding: 5px; border: 1px solid black; z-index: 1000;">
            <p style="margin: 0; font-size: 12px;">
                Note: This choropleth map visualizes the Subsidized Housing Coverage Ratio, defined as the ratio of 
                subsidized housing units to low-income renter households per census tract, classified into bins: 
                [0–0.2], [0.2–0.4], [0.4–0.6], [0.6–0.8], and >0.8. A low ratio indicates a limited number of 
                subsidized units relative to the estimated low-income renter households, suggesting high potential 
                competition for available subsidized housing.
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(footnote))
        
        # Add LayerControl (optional, since only one layer)
        folium.LayerControl().add_to(m)
        
        # Save the map
        m.save('data/housing_coverage_choropleth.html')
        
        # Return the map
        return m

def create_months_to_expiration_choropleth(gdf, output_path='data/months_to_expiration_choropleth.html'):
    """
    Create a Folium choropleth map for average months to expiration by census tract.
    
    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame with columns ['CensusTract', 'expiration', 
      'AvgMonthsToExpire', 'TotalUnits', 'geometry']
    - output_path (str): Path to save the output HTML file
    
    Returns:
    - None: Saves the map to the specified output path
    """
    # Input validation
    required_columns = ['CensusTract', 'expiration', 'AvgMonthsToExpire', 'TotalUnits', 'geometry']
    if not all(col in gdf.columns for col in required_columns):
        raise ValueError(f"GeoDataFrame must contain columns: {required_columns}")
    
    # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
    if gdf.crs is None:
        print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
        gdf = gdf.set_crs(epsg=4326)
    
    # Reproject to a projected CRS (Albers Equal Area, EPSG:5070) for centroid calculation
    gdf_projected = gdf.to_crs(epsg=5070)
    
    # Calculate the centroid in the projected CRS
    centroid_projected = gdf_projected.geometry.centroid
    
    # Reproject the centroid back to EPSG:4326 for Folium
    centroid = centroid_projected.to_crs(epsg=4326)
    map_center = [centroid.y.mean(), centroid.x.mean()]
    
    # Reproject the GeoDataFrame to EPSG:4326 for Folium
    gdf = gdf.to_crs(epsg=4326)
    
    # Initialize the Folium map
    m = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")
    
    # Filter data for 5-plus-year and 5-year expirations
    gdf_5_plus_yr = gdf[gdf['expiration'] == 'expiration beyond 5 years'].copy()
    gdf_5yr = gdf[gdf['expiration'] == 'expiration within 5 years'].copy()
    
    # Create a formatted column for tooltips
    gdf_5_plus_yr['AvgMonthsToExpireFormatted'] = gdf_5_plus_yr['AvgMonthsToExpire'].apply(lambda x: f"{x:.1f}")
    gdf_5yr['AvgMonthsToExpireFormatted'] = gdf_5yr['AvgMonthsToExpire'].apply(lambda x: f"{x:.1f}")
    
    # Create color scales
    # 5-plus-year expiration: Dark blue (low months) to light blue (high months, capped at 600)
    colormap_10yr = cm.LinearColormap(
        colors=['#00008B', '#ADD8E6'],  # Dark blue to light blue
        vmin=gdf_5_plus_yr['AvgMonthsToExpire'].min() if not gdf_5_plus_yr.empty else 60,
        vmax=600,  # Cap at 600 months
        caption='Average Months to Expiration (5 Plus Year)'
    )
    
    # 5-year expiration: Dark red (low months) to light red (high months)
    colormap_5yr = cm.LinearColormap(
        colors=['#8B0000', '#FFB6C1'],  # Dark red to light red
        vmin=gdf_5yr['AvgMonthsToExpire'].min() if not gdf_5yr.empty else 0,
        vmax=gdf_5yr['AvgMonthsToExpire'].max() if not gdf_5yr.empty else 60,
        caption='Average Months to Expiration (5-Year)'
    )
    
    # Add 5-plus-year expiration layer using GeoJson
    if not gdf_5_plus_yr.empty:
        style_function_10yr = lambda x: {
            'fillColor': colormap_10yr(min(x['properties']['AvgMonthsToExpire'], 600)) 
                         if x['properties']['AvgMonthsToExpire'] else '#ffffff',
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.7
        }
        
        folium.GeoJson(
            gdf_5_plus_yr,
            style_function=style_function_10yr,
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'AvgMonthsToExpireFormatted', 'TotalUnits'],
                aliases=['Census Tract', 'Avg Months to Expire', 'Total Units'],
                localize=True,
                labels=True
            ),
            name='5-Plus Year Expiration'
        ).add_to(m)
        
        # Add colormap to map
        colormap_10yr.add_to(m)
    
    # Add 5-year expiration layer using GeoJson
    if not gdf_5yr.empty:
        style_function_5yr = lambda x: {
            'fillColor': colormap_5yr(x['properties']['AvgMonthsToExpire']) 
                         if x['properties']['AvgMonthsToExpire'] else '#ffffff',
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.7
        }
        
        folium.GeoJson(
            gdf_5yr,
            style_function=style_function_5yr,
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'AvgMonthsToExpireFormatted', 'TotalUnits'],
                aliases=['Census Tract', 'Avg Months to Expire', 'Total Units'],
                localize=True,
                labels=True
            ),
            name='5-Year Expiration'
        ).add_to(m)
        
        # Add colormap to map
        colormap_5yr.add_to(m)
    
    # Add footnote using custom HTML div
    footnote_html = """
    <div style="position: fixed; 
                bottom: 10px; 
                left: 10px; 
                z-index: 9999; 
                font-size: 12px; 
                background-color: rgba(255, 255, 255, 0.8); 
                padding: 5px; 
                border-radius: 3px;">
        Note: More than 600 months (50 years) is considered perpetual subsidy and is capped in the legend.
    </div>
    """
    m.get_root().html.add_child(folium.Element(footnote_html))
    
    # Add LayerControl to toggle between layers
    folium.LayerControl().add_to(m)
    
    # Save the map
    m.save(output_path)
    print(f"Map saved to {output_path}")

def plot_risk_distribution_by_state(df: pd.DataFrame, 
                                     save_path: str = 'risk_distribution_by_state.png'):
    """
    Plots a stacked bar chart showing the percentage of units by risk category for each state.
    Filters out unknown categories. Legend is placed outside the plot.
    """

    # --- Step 1: Filter & Aggregate ---
    df_clean = df[df['ConditionRiskCategory'] != 'Unknown (No Score)'].copy()
    df_clean = df_clean[df_clean['State'].notna()]
    
    df_grouped = df_clean.groupby(['State', 'ConditionRiskCategory'])['TotalUnits'].sum().reset_index()
    df_total = df_grouped.groupby('State')['TotalUnits'].sum().reset_index().rename(columns={'TotalUnits': 'TotalPerState'})
    df_merged = pd.merge(df_grouped, df_total, on='State')
    df_merged['Percent'] = df_merged['TotalUnits'] / df_merged['TotalPerState'] * 100

    # --- Step 2: Prepare data for plotting ---
    category_order = [
        'Low Risk (Score >= 80)',
        'Moderate Risk (Score 60-79)',
        'Concern (H&S Non-Life-Threatening)',
        'High Risk (Score < 60)',
        'Urgent (H&S Life-Threatening)'
    ]
    color_map = {
        'Low Risk (Score >= 80)': '#a6cee3',
        'Moderate Risk (Score 60-79)': '#1f78b4',
        'Concern (H&S Non-Life-Threatening)': '#ff7f00',
        'High Risk (Score < 60)': '#e31a1c',
        'Urgent (H&S Life-Threatening)': '#6a3d9a'
    }

    df_plot = df_merged.pivot_table(index='State', columns='ConditionRiskCategory',
                                    values='Percent', fill_value=0)
    df_plot = df_plot[category_order]

    state_order = df_clean.groupby('State')['TotalUnits'].sum().sort_values(ascending=False).index.tolist()
    df_plot = df_plot.loc[state_order]

    # --- Step 3: Plotting ---
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot) * 0.4), 6))

    bottom = pd.Series([0] * df_plot.shape[0], index=df_plot.index)
    for category in category_order:
        heights = df_plot[category]
        ax.bar(df_plot.index, heights, bottom=bottom,
               color=color_map[category], label=category)
        bottom += heights

    # --- Step 4: Labels and Layout ---
    ax.set_ylabel('Percentage of Units', fontsize=12)
    ax.set_xlabel('State', fontsize=12)
    ax.set_title('Distribution of Percentage of Subsidized Units by Property Condition Risk Category and State',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(len(df_plot.index)))
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Legend outside the plot
    ax.legend(title='Condition Risk Category', bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=9, title_fontsize=10)

    plt.subplots_adjust(right=0.78)  # create space for the legend
    plt.tight_layout()

    # --- Step 5: Save ---
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {save_path}")





def create_risk_choropleth(gdf):
    # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
    if gdf.crs is None:
        print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
        gdf = gdf.set_crs(epsg=4326)
    
    # Reproject to a projected CRS (Albers Equal Area, EPSG:5070) for centroid calculation
    gdf_projected = gdf.to_crs(epsg=5070)
    
    # Calculate the centroid in the projected CRS
    centroid_projected = gdf_projected.geometry.centroid
    
    # Reproject the centroid back to EPSG:4326 for Folium
    centroid = centroid_projected.to_crs(epsg=4326)
    map_center = [centroid.y.mean(), centroid.x.mean()]
    
    # Reproject the GeoDataFrame to EPSG:4326 for Folium
    gdf = gdf.to_crs(epsg=4326)
    
    # Initialize the Folium map
    m = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")
    
    # Define categories (excluding 'Unknown (No Score)')
    categories = [
        'Low Risk (Score >= 80)',
        'Moderate Risk (Score 60-79)',
        'Concern (H&S Non-Life-Threatening)',
        'High Risk (Score < 60)',
        'Urgent (H&S Life-Threatening)'
    ]

    # Define colors for each category (matching the choropleth map)
    color_schemes = {
        'Low Risk (Score >= 80)': ['#006400', '#90EE90'],  # Light green
        'Moderate Risk (Score 60-79)': ['#FFA500', '#FFFFE0'],  # Light yellow
        'Concern (H&S Non-Life-Threatening)': ['#8B0000', '#FFB6C1'],  # Light red
        'High Risk (Score < 60)': ['#4B0082', '#E6E6FA'],  # Light purple
        'Urgent (H&S Life-Threatening)': ['#1E90FF', '#ADD8E6'] # Light blue
    }

    
    # Create a formatted column for tooltips
    gdf['AvgMonthsToExpireFormatted'] = gdf['AvgMonthsToExpire'].apply(lambda x: f"{x:.1f}")
    
    # Dictionary to store colormaps for each category
    colormap_dict = {}
    
    # Create colormaps for each category
    for category in categories:
        gdf_category = gdf[gdf['ConditionRiskCategory'] == category].copy()
        if not gdf_category.empty:
            colormap = cm.LinearColormap(
                colors=color_schemes[category],
                vmin=0,  # Fixed min value
                vmax=50,  # Fixed max value (as per footnote)
                caption=f'Average Months to Expiration ({category})'
            )
            colormap_dict[category] = colormap
    
    # Add a layer for each category using FeatureGroup for better layer control
    for category in categories:
        # Filter GeoDataFrame for the current category
        gdf_category = gdf[gdf['ConditionRiskCategory'] == category].copy()
        
        if not gdf_category.empty:
            # Create a FeatureGroup for this category
            fg = folium.FeatureGroup(name=category, show=True)
            
            # Get the colormap for this category
            colormap = colormap_dict[category]
            
            # Define style function
            style_function = lambda x, colormap=colormap: {
                'fillColor': colormap(x['properties']['AvgMonthsToExpire']) if x['properties']['AvgMonthsToExpire'] is not None else '#ffffff',
                'color': 'black',
                'weight': 0.1,
                'fillOpacity': 0.7
            }
            
            # Add GeoJson layer to the FeatureGroup
            folium.GeoJson(
                gdf_category,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['CensusTract', 'AvgMonthsToExpireFormatted', 'TotalUnits'],
                    aliases=['Census Tract', 'Avg Months to Expire', 'Total Units'],
                    localize=True,
                    labels=True
                ),
            ).add_to(fg)
            
            # Add the FeatureGroup to the map
            fg.add_to(m)
            
            # Add colormap to map (positioned separately to avoid overlap)
            colormap.add_to(m)
    
    # Add LayerControl to toggle between layers
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add footnote at the bottom using custom HTML
    footnote = """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); 
                background-color: white; padding: 5px; border: 1px solid black; z-index: 1000;">
        <p style="margin: 0; font-size: 12px;">
            <b>NOTE:</b> This visualization includes 507 properties with subsidy expiration dates beyond the year 2100, 
            likely indicative of perpetual or long-term subsidy agreements, and 630 properties with subsidies 
            expired prior to April 16, 2025. For cartographic purposes, Avg Month to Expire duration is capped at 0 to 50 months.
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(footnote))
    
    # Save the map
    m.save('data/risk_choropleth.html')


def plot_subsidized_units_expiry_by_state(df, save_path=None):
    """
    Plot subsidized units with expiring subsidies by state.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'State' and 'TotalUnits' columns
    save_path (str, optional): File path to save the figure (e.g., 'output.png'). If None, displays the plot.
    """
    # Get today's date and calculate the expiration year (5 years from today)
    run_date = datetime.today()
    expiration_date = run_date.replace(year=run_date.year + 5).strftime('%B %d, %Y')
    
    # Set plot style
    plt.style.use('seaborn-v0_8')  
    plt.figure(figsize=(12, 9), dpi=300) 
    bar_width = 0.4
    
    # Create a custom color palette that transitions from dark purple to yellow
    colors = sns.color_palette(
        "blend:#4B0082,#1E90FF,#20B2AA,#32CD32,#FFFF00",  # Purple, Blue, Teal, Green, Yellow
        n_colors=len(df['State'])
    )
    
    # Sort the DataFrame by TotalUnits in descending order to match the chart's ordering
    df_sorted = df.sort_values(by='TotalUnits', ascending=False)
    
    # Plot the bars with the custom color palette
    bars = plt.bar(df_sorted['State'], df_sorted['TotalUnits'], 
                   color=colors, edgecolor='grey', linewidth=0.5)
    
    # Labeling axes and title
    plt.xlabel('State', fontsize=12, labelpad=10, fontweight='bold')
    plt.ylabel('Total Units at Risk', fontsize=12, labelpad=10, fontweight='bold')
    plt.title('Subsidized Units with Expiring Subsidies by State (2025-2030)', 
              fontsize=14, pad=15, fontweight='bold')
    
    # Rotate x-tick labels to prevent overlap
    plt.xticks(rotation=90, ha='center', fontsize=10)
    
    # Add grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to ensure tight fit and avoid clipping
    plt.tight_layout()

    # Save or display with higher DPI for better upscaling
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    # Close figure
    plt.close()

def plot_rent_ratio_faceted_bubble_enhanced(df: pd.DataFrame) -> sns.axisgrid.FacetGrid:
    # --- Create FacetGrid with custom figure size ---
    g = sns.relplot(
        data=df,
        x="State",
        y="RentRatio",
        size="TotalUnits",
        col="expiration",
        col_order=['expiration within 5 years', 'expiration beyond 5 years'],
        kind="scatter",
        col_wrap=2,
        sizes=(100, 800),
        alpha=0.7,
        height=6,         # Slightly taller
        aspect=1.8,       # Wider to prevent label overlap
        legend=False,
        palette="viridis"
    )

    # --- Add Text Labels ---
    for i, facet_name in enumerate(['expiration within 5 years', 'expiration beyond 5 years']):
        data_subset = df[df['expiration'] == facet_name]
        ax = g.axes.flat[i]
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, linestyle='--', alpha=0.5)
        for _, row in data_subset.iterrows():
            ax.text(
                x=row['State'],
                y=row['RentRatio'],
                s=f"{row['RentRatio']:.2f}",
                fontsize=8,
                color='black',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
            )

    # --- Customize Axes ---
    g.set_xticklabels(rotation=90, fontsize=10)
    g.set_axis_labels("State", "Rent Ratio (Median Gross Rent/Fair Market Rent)", fontsize=12)

    # --- Adjust layout to make space for titles ---
    g.fig.subplots_adjust(top=0.90, bottom=0.10)

    # --- Add Title ---
    g.fig.suptitle(
        "Rent Ratio Comparison by State and Contract Expiration Status",
        fontsize=16,
        weight='bold',
        ha='center'
    )

    # --- Add Footnote ---
    g.fig.text(
        x=0.01,
        y=0.02,
        s="*Bubble size represents the average TotalUnits for the corresponding State and Expiration category.",
        ha='left',
        va='bottom',
        fontsize=9,
        color='gray'
    )

    # --- Save Final Plot ---
    g.figure.savefig("rent_ratio_plot_fixed.png", dpi=300, bbox_inches="tight")
    print("Saved plot as 'rent_ratio_plot_fixed.png'")

    return g


if __name__ == '__main__':

    # NHPD stands for the National Housing Preservation Database. It helps identify properties at risk of losing their affordability due to exp≥iring subsidies, helping prevent displacement and preserve affordable housing.

    subsidized_df = pd.read_excel('data/Active and Inconclusive Properties.xlsx')

    # Idea: Predicting and Preventing the loss of Affordable Housing: Risk Assessment of Subsidized Properties

    # Objective: Develop a risk assessment framework to identify subsidized housing properties at greatest risk of losing their affordability status due to expiring subsidies. 

    subsidized_df['EarliestEndDate'] = pd.to_datetime(subsidized_df['EarliestEndDate'], errors='coerce')
    subsidized_df['LatestEndDate'] = pd.to_datetime(subsidized_df['LatestEndDate'], errors='coerce')
    
    # consider only active property status (properties actively receiving subsidies)
    subsidized_df_v1  = subsidized_df[subsidized_df['PropertyStatus'] == 'Active']

    # consider non-missing latest end date
    subsidized_df_v1 = subsidized_df_v1[subsidized_df_v1['LatestEndDate'].notna()]
    
    print(len(subsidized_df_v1))
    print(len(subsidized_df_v1.dropna(subset=['CensusTract'])))
    subsidized_df_v1 = subsidized_df_v1.dropna(subset=['CensusTract']) # dropped (missing census tract)
    subsidized_df_v1['CensusTract'] = subsidized_df_v1['CensusTract'].astype(int).astype(str).str.zfill(11)


    # visualizing properties expiration within 5 years and 10 years from today
    current_date = datetime(2025, 4, 16) # fix current date to april 15, 2025


    subsidized_df_v1['MonthsToExpire'] = ((subsidized_df_v1['LatestEndDate'] - current_date).dt.days / 30.44).round(1)

    # Time windows for expiration analysis
    five_years = current_date + pd.DateOffset(years=5)
    ten_years = current_date + pd.DateOffset(years=10)


    conditions = [
    (subsidized_df_v1['LatestEndDate'] >= current_date) & (subsidized_df_v1['LatestEndDate'] <= five_years),
    (subsidized_df_v1['LatestEndDate'] > five_years)
]

    choices = ['expiration within 5 years', 'expiration beyond 5 years']
    subsidized_df_v1['expiration'] = np.select(conditions, choices, default='expired or unknown')



    expiring_5yr = subsidized_df_v1[(subsidized_df_v1['LatestEndDate'] >= current_date) & (subsidized_df_v1['LatestEndDate'] <= five_years)].copy()

    # Calculate total units at risk
    units_5yr = expiring_5yr['TotalUnits'].sum()

    summary_5yr_by_state = expiring_5yr.groupby(['State']).agg({
    'TotalUnits': 'sum',
    'NHPDPropertyID': 'count'  # Number of properties
}).reset_index()

    
    print("Subsidy Expiration Snapshot (as of April 16, 2025):")
    print(f"Total units with subsidies expiring within 5 years (by {five_years.date()}): {units_5yr}")

    print("\nUnits at risk within 5 years by State:")
    print(summary_5yr_by_state)


    print("\nUnits at risk within 5 years by State (including number of properties):")
    print(summary_5yr_by_state[['State', 'TotalUnits', 'NHPDPropertyID']])

    plot_subsidized_units_expiry_by_state(summary_5yr_by_state)


    # merge to census tract dataset
    census_df = pd.read_parquet("data/acs5_2023_tract_v1_data_multi_state.parquet", engine="fastparquet")
    census_df['GEOID'] = census_df['STATE']+census_df['COUNTY']+census_df['TRACT']
    census_df['CensusTract'] = census_df['GEOID'].astype(str).str.zfill(11)

    
    df_subsidized_with_acs = pd.merge(subsidized_df_v1, census_df, on='CensusTract', how='left')
   
    
    df_subsidized_with_acs['geometry'] = df_subsidized_with_acs['geometry'].apply(lambda x: wkb.loads(x, hex=False) if pd.notnull(x) else None)
    df_subsidized_with_acs.to_csv("data/active_subsidized_with_acs.csv",index = False)

    ## calculate days to expire by census tract (use total units instead)
    days_to_expire_by_tract_df = (
    df_subsidized_with_acs
    .groupby(['CensusTract', 'expiration'])[['MonthsToExpire', 'TotalUnits']]
    .agg({'MonthsToExpire': 'mean', 'TotalUnits': 'sum'})
    .reset_index()
    .rename(columns={'MonthsToExpire': 'AvgMonthsToExpire'})
)
    days_to_expire_by_tract_df = days_to_expire_by_tract_df[days_to_expire_by_tract_df['expiration']!='others']

    days_to_expire_by_tract_df = days_to_expire_by_tract_df.merge(df_subsidized_with_acs[['CensusTract', 'geometry']].drop_duplicates(subset = ['CensusTract']), on='CensusTract', how='left')
    days_to_expire_by_tract_gdf = gpd.GeoDataFrame(days_to_expire_by_tract_df, geometry='geometry')
    


    ## Folium Map : Census Tract with 5-year (Red) and 10-year (Blue) expiration Heatmap 
    
    create_months_to_expiration_choropleth(days_to_expire_by_tract_gdf)



    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Folium map Risk Score Categorization : Census Tract with 5-year (Red) and 10-year (Blue) expiration Heatmap 

    def parse_reac_score(score_str):
        if pd.isna(score_str):
            return np.nan, np.nan
        score_str = str(score_str).strip()
        # Use regex to find numeric part and optional trailing letter (a, b, c)
        match = re.match(r'^(\d+)([abc]?)$', score_str, re.IGNORECASE)
        if match:
            numeric_score = int(match.group(1))
            qualifier = match.group(2).lower() if match.group(2) else None
            # Ensure score is within valid range (optional, but good practice)
            if 0 <= numeric_score <= 100:
                return numeric_score, qualifier
            else: # Handle potential invalid numbers if they exist
                return np.nan, np.nan
        else: # Handle cases that don't match the pattern
            return np.nan, np.nan

    df_property_risk_score = df_subsidized_with_acs.copy() 


    for i in [1, 2, 3]:
        df_property_risk_score[[f'ReacScore{i}_Numeric', f'ReacScore{i}_Qualifier']] = df_property_risk_score[f'ReacScore{i}'].apply(lambda x: pd.Series(parse_reac_score(x)))
        # Ensure date columns are datetime objects (important!)
        df_property_risk_score[f'ReacScore{i}Date'] = pd.to_datetime(df_property_risk_score[f'ReacScore{i}Date'], errors='coerce')

    # --- 3. Create 'Most Recent' Score/Date/Qualifier Columns ---
    # Use np.select for conditional assignment based on availability, prioritizing 1 -> 2 -> 3
    conditions = [
        df_property_risk_score['ReacScore1_Numeric'].notna(),
        df_property_risk_score['ReacScore2_Numeric'].notna(),
        df_property_risk_score['ReacScore3_Numeric'].notna()
    ]

    choices_score = [
        df_property_risk_score['ReacScore1_Numeric'],
        df_property_risk_score['ReacScore2_Numeric'],
        df_property_risk_score['ReacScore3_Numeric']
    ]
    choices_qualifier = [
        df_property_risk_score['ReacScore1_Qualifier'],
        df_property_risk_score['ReacScore2_Qualifier'],
        df_property_risk_score['ReacScore3_Qualifier']
    ]
    choices_date = [
        df_property_risk_score['ReacScore1Date'],
        df_property_risk_score['ReacScore2Date'],
        df_property_risk_score['ReacScore3Date']
    ]

    df_property_risk_score['MostRecentReacScore'] = np.select(conditions, choices_score, default=np.nan)
    df_property_risk_score['MostRecentReacQualifier'] = np.select(conditions, choices_qualifier, default=None) # Use None or np.nan for missing qualifiers
    df_property_risk_score['MostRecentReacDate'] = np.select(conditions, choices_date, default=pd.NaT)

    # --- 4. Define Risk Categories (Example) ---
    # You need to refine these based on specific program rules or analysis goals
    def assign_risk(row):
        score = row['MostRecentReacScore']
        qualifier = row['MostRecentReacQualifier']
        date = row['MostRecentReacDate']

        if pd.isna(score):
            return "Unknown (No Score)"

        # Sequential evaluation following the prioritized order in the documentation
        # (i) Urgent: H&S Life-Threatening ('c' Flag)
        if qualifier == 'c':
            return "Urgent (H&S Life-Threatening)"
        
        # (ii) High Risk: Score < 60 (Score-Based)
        if score < 60:
            return "High Risk (Score < 60)"
        
        # (iii) Concern: H&S Non-Life-Threatening ('b' Flag)
        if qualifier == 'b':
            return "Concern (H&S Non-Life-Threatening)"
        
        # (iv) Moderate Risk: Score 60-79 (Score-Based)
        if 60 <= score < 80:
            return "Moderate Risk (Score 60-79)"
        
        # (v) Low Risk: Score >= 80 (Score-Based)
        if score >= 80:
            return "Low Risk (Score >= 80)"
        
        # Catch-all for unexpected cases
        return "Unknown (Other)"

    df_property_risk_score['ConditionRiskCategory'] = df_property_risk_score.apply(assign_risk, axis=1)

    # --- 5. Analyze ---
    print(df_property_risk_score['ConditionRiskCategory'].value_counts(dropna=False))

    
    property_condtion_risk_score_with_expiration_by_tract_df = (
        df_property_risk_score.groupby(['CensusTract','ConditionRiskCategory'])[['MonthsToExpire', 'TotalUnits']]
        .agg({'MonthsToExpire': 'mean', 'TotalUnits': 'sum'})
    ).reset_index().rename(columns={'MonthsToExpire': 'AvgMonthsToExpire'})


    property_condtion_risk_score_with_expiration_by_tract_df = property_condtion_risk_score_with_expiration_by_tract_df[property_condtion_risk_score_with_expiration_by_tract_df['ConditionRiskCategory']!= 'Unknown (No Score)']
   
    
    property_condtion_risk_score_with_expiration_by_tract_df = property_condtion_risk_score_with_expiration_by_tract_df.merge(df_subsidized_with_acs[['CensusTract', 'geometry']].drop_duplicates(subset = ['CensusTract']), on='CensusTract', how='left')
    property_condtion_risk_score_with_expiration_by_tract_gdf = gpd.GeoDataFrame(property_condtion_risk_score_with_expiration_by_tract_df, geometry='geometry')
    

    df_property_risk_score_grouped = df_property_risk_score.groupby(['State', 'ConditionRiskCategory']).agg({'TotalUnits':'sum'}).reset_index()

    create_risk_category_barchart(df_property_risk_score_grouped)

    plot_risk_distribution_by_state(df_property_risk_score_grouped)

    create_risk_choropleth(property_condtion_risk_score_with_expiration_by_tract_gdf)


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    df_housing_gap = df_subsidized_with_acs.copy() 
    df_housing_gap_v1 = df_housing_gap[['CensusTract','TotalUnits' ,'Renter_HH_Income_Less_5K', 'Renter_HH_Income_5K_10K', 'Renter_HH_Income_10K_15K', 'Renter_HH_Income_15K_20K', 'Renter_HH_Income_20K_25K', 'Renter_HH_Income_25K_35K', 'Renter_HH_Income_35K_50K']]
    df_housing_gap_v1= df_housing_gap_v1.dropna()
    df_housing_gap_v1['Total_LowIncome_Renter_HH'] = df_housing_gap_v1[
    ['Renter_HH_Income_Less_5K', 'Renter_HH_Income_5K_10K', 'Renter_HH_Income_10K_15K',
     'Renter_HH_Income_15K_20K', 'Renter_HH_Income_20K_25K', 'Renter_HH_Income_25K_35K',
     'Renter_HH_Income_35K_50K']
].sum(axis=1)


    def calculate_ratio(numerator, denominator, scale=1):
        if pd.isna(denominator) or denominator <= 0:
            return np.nan # Or None, or 0 depending on desired handling
        else:
            return (numerator / denominator) * scale

    subsidied_housing_coverage_df = df_housing_gap_v1.groupby('CensusTract').agg({'Total_LowIncome_Renter_HH':'first', 'TotalUnits':'sum'}).reset_index()
    subsidied_housing_coverage_df['Subsidized_Housing_Coverage_Ratio'] = subsidied_housing_coverage_df.apply(
    lambda row: calculate_ratio(row['TotalUnits'], row['Total_LowIncome_Renter_HH'], scale=1),
    axis=1
)   


    # Box plot to visualize the distribution of the Subsidized Housing Coverage Ratio

    

    plot_subsidized_housing_boxplot(subsidied_housing_coverage_df['Subsidized_Housing_Coverage_Ratio'] )


    


    subsidied_housing_coverage_df = subsidied_housing_coverage_df.merge(df_subsidized_with_acs[['CensusTract', 'geometry']].drop_duplicates(subset = ['CensusTract']), on='CensusTract', how='left')
    subsidied_housing_coverage_gdf = gpd.GeoDataFrame(subsidied_housing_coverage_df, geometry='geometry')

    create_housing_coverage_choropleth(subsidied_housing_coverage_gdf)


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Market Rent Pressure
    market_rent_pressure_df = df_subsidized_with_acs.copy()
    market_rent_pressure_df_v1 = market_rent_pressure_df[['CensusTract','Median_Gross_Rent','FairMarketRent_2BR', 'expiration', 'TotalUnits']]

    market_rent_pressure_df_v1 = market_rent_pressure_df_v1.dropna()
    market_rent_pressure_df_grp = market_rent_pressure_df_v1.groupby('CensusTract').agg({'Median_Gross_Rent':'mean','FairMarketRent_2BR':'mean', 'TotalUnits':'sum'}).reset_index()

    
    market_rent_pressure_df_grp['RentRatio'] = market_rent_pressure_df_grp['Median_Gross_Rent'] / market_rent_pressure_df_grp['FairMarketRent_2BR']
    
    plot_rent_ratio_boxplot(market_rent_pressure_df_grp['RentRatio'], title="Boxplot of Rent Ratio", ylabel="Rent Ratio", figsize=(14, 8))

    market_rent_pressure_df_grp_v1 = market_rent_pressure_df_grp.merge(df_subsidized_with_acs[['CensusTract', 'geometry', 'expiration', 'State']].drop_duplicates(subset = ['CensusTract']), on='CensusTract', how='left')
    market_rent_pressure_df_grp_gdf = gpd.GeoDataFrame(market_rent_pressure_df_grp_v1, geometry='geometry')
    
    create_rent_ratio_choropleth(market_rent_pressure_df_grp_gdf)
        
    
    # Group by expiration & state and calculate mean RentRatio and TotalUnits

    rent_ratio_by_state_expiration_df = market_rent_pressure_df_grp_v1.groupby(["State","expiration"]).agg({'RentRatio':'mean','TotalUnits':'mean'}).reset_index()

    rent_ratio_by_state_expiration_df = rent_ratio_by_state_expiration_df[rent_ratio_by_state_expiration_df['expiration']!= 'expired or unknown']


    
    def plot_rent_ratio_scatter(df, save_path=None):
        """
        Generate a faceted scatter plot for RentRatio by state, with bubble size encoding TotalUnits,
        excluding 'expired or unknown' expiration status.

        Parameters:
        - df (pd.DataFrame): DataFrame with columns ['State', 'expiration', 'RentRatio', 'TotalUnits']
        - save_path (str, optional): File path to save the figure (e.g., 'output.png'). If None, displays the plot.

        Returns:
        - None: Displays or saves the plot.
        """
        # Input validation
        required_columns = ['State', 'expiration', 'RentRatio', 'TotalUnits']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Filter out 'expired or unknown'
        df_filtered = df[df['expiration'].isin(['expiration beyond 5 years', 'expiration within 5 years'])].copy()

        # Normalize TotalUnits for bubble size (scale to 200–1300 for larger bubbles)
        max_units = df_filtered['TotalUnits'].max()
        min_units = df_filtered['TotalUnits'].min()
        if max_units > min_units:
            df_filtered['BubbleSize'] = 200 + 1100 * (df_filtered['TotalUnits'] - min_units) / (max_units - min_units)
        else:
            df_filtered['BubbleSize'] = 600  # fallback if all units are the same

        # Set Seaborn style
        sns.set_style("whitegrid")

        # Create faceted plot
        g = sns.FacetGrid(
            df_filtered,
            row='expiration',  # Vertical layout
            height=4,
            aspect=2,
            sharey=True,
            gridspec_kws={'hspace': 0.3}
        )

        # Plot scatter with variable bubble size
        g.map_dataframe(
            sns.scatterplot,
            x='State',
            y='RentRatio',
            size='BubbleSize',
            color='#1f77b4',
            alpha=0.6,
            legend=False
        )

        # Customize axes
        g.set_titles('{row_name}', fontsize=12, fontweight='bold')
        g.set_xlabels('State', fontsize=11)
        g.set_ylabels('Rent Ratio\n(Median Gross Rent / Fair Market Rent, 1.0)', fontsize=11)

        # Rotate x-axis labels for better readability
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=9, rotation=90)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_ylim(0.8, 1.4)

        # Add the main title with adjusted position to ensure visibility
        g.fig.suptitle('Rent Ratio Comparison By State and Subsidy Expiration Status', fontsize=18, y=0.95)

        # Add a note about bubble size
        plt.figtext(0.5, 0.01, 'Bubble size indicates TotalUnits per State and Expiration category',
                    ha='center', fontsize=11, style='italic')

        # Adjust layout to prevent title overlap
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        # Save or display with higher DPI for better upscaling
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        # Close figure
        plt.close()

