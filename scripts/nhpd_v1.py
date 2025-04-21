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


# Ensure the output directory for charts exists
os.makedirs('data/charts', exist_ok=True)

def plot_rent_ratio_boxplot(data, title="Boxplot of Rent Ratio", ylabel="Rent Ratio", figsize=(10, 8), save_path='data/charts/rent_ratio_boxplot.png'):
        """
        Create a boxplot for RentRatio data, save it, and make text larger.

        Parameters:
        - data: Pandas Series or array-like, containing RentRatio values.
        - title: String, title of the plot (default: "Boxplot of Rent Ratio").
        - ylabel: String, label for y-axis (default: "Rent Ratio").
        - figsize: Tuple, figure size as (width, height) (default: (10, 8)).
        - save_path: String, path to save the plot image.

        Returns:
        - Displays a boxplot.
        """
        # Set the style for better aesthetics
        sns.set_style("whitegrid")

        # Create figure and axis
        plt.figure(figsize=figsize)

        # Plot boxplot using Seaborn
        sns.boxplot(y=data, color="skyblue", width=0.4)

        # Customize plot with larger fonts
        plt.title(title, fontsize=18, pad=15)
        plt.ylabel(ylabel, fontsize=16, labelpad=10)
        plt.yticks(fontsize=12)
        plt.xticks([])  # Remove x-axis ticks since it's a single boxplot

        # Add mean line
        mean_val = np.mean(data)
        plt.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")
        plt.legend(fontsize=14)

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot with high resolution
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

        # Show plot
        plt.show()


def create_rent_ratio_choropleth(gdf, rent_ratio_col='RentRatio',
                                 output_file='data/rent_ratio_choropleth.html'):
        """
        Create an interactive Folium choropleth map of RentRatio from a GeoDataFrame.

        Parameters:
        - gdf: GeoDataFrame, contains RentRatio and geometry columns.
        - rent_ratio_col: String, name of the RentRatio column (default: 'RentRatio').
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
        elif gdf.crs.to_epsg() != 4326:
             gdf = gdf.to_crs(epsg=4326) # Reproject if not already EPSG:4326


        # Calculate map center robustly (handle potential empty geometries)
        if gdf.geometry.is_empty.any():
            print("Warning: GeoDataFrame contains empty geometries. Centroid calculation might be affected.")
            gdf_valid_geom = gdf[~gdf.geometry.is_empty]
            if gdf_valid_geom.empty:
                map_center = [39.8283, -98.5795] # Default center (US) if no valid geometries
                print("Warning: No valid geometries found. Using default map center.")
            else:
                centroid_projected = gdf_valid_geom.to_crs(epsg=5070).geometry.centroid
                centroid = centroid_projected.to_crs(epsg=4326)
                map_center = [centroid.y.mean(), centroid.x.mean()]
        else:
            centroid_projected = gdf.to_crs(epsg=5070).geometry.centroid
            centroid = centroid_projected.to_crs(epsg=4326)
            map_center = [centroid.y.mean(), centroid.x.mean()]


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
            include_lowest=True,
            right=False # Ensure 0.8 is included in '>0.8'
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
            index=[0, 0.2, 0.4, 0.6, 0.8, gdf[rent_ratio_col].max() if not gdf.empty else 1.0], # Use actual max for upper bound if available
            vmin=0,
            vmax=gdf[rent_ratio_col].max() if not gdf.empty else 1.0, # Adjust vmax based on data
            caption='Rent Ratio'
        )

        # Format RentRatio for tooltips
        gdf['RentRatioFormatted'] = gdf[rent_ratio_col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

        # Define style function
        def style_function(feature):
            bin_label = feature['properties'].get('RentRatioBin') # Use .get for safety
            fill_color = color_map.get(bin_label, '#808080') # Default to gray if bin is missing or NaN
            # Handle NaN RentRatio values specifically
            if pd.isna(feature['properties'].get(rent_ratio_col)):
                fill_color = '#808080' # Gray for NaN

            return {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 0.1,
                'fillOpacity': 0.7 if bin_label else 0.3 # Lower opacity if no bin
            }

        # Prepare tooltip fields (CensusTract first, then RentRatio)
        tooltip_fields = []
        tooltip_aliases = []
        if 'CensusTract' in gdf.columns:
            tooltip_fields.append('CensusTract')
            tooltip_aliases.append('Census Tract:')
        tooltip_fields.append('RentRatioFormatted')
        tooltip_aliases.append('Rent Ratio:')

        # Add GeoJson layer
        geojson_layer = folium.GeoJson(
            gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=True, # Make tooltip stay until next hover
                labels=True,
                style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;")
            ),
            name='Rent Ratio by Census Tract',
            highlight_function=lambda x: {'weight': 2, 'color': 'yellow', 'fillOpacity': 0.8} # Highlight on hover
        ).add_to(m)

        # Add colormap to map
        colormap.add_to(m)

        # Add footnote at the bottom using custom HTML
        footnote = """
        <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                    background-color: rgba(255, 255, 255, 0.85); padding: 8px; border: 1px solid grey; border-radius: 5px; z-index: 1000; max-width: 80%;">
            <p style="margin: 0; font-size: 13px; text-align: center;">
                <b>Rent Ratio:</b> Median Gross Rent / Fair Market Rent (2BR). <b>From an affordability perspective:</b> Lower is better (indicates less rent pressure). Gray areas indicate missing data.
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(footnote))

        # Add LayerControl
        folium.LayerControl().add_to(m)

        # Save the map
        m.save(output_file)
        print(f"Choropleth map saved to {output_file}")

        # Return the map
        return m

def create_housing_coverage_choropleth(gdf, output_file='data/housing_coverage_choropleth.html'):
        """
        Create an interactive Folium choropleth map of Subsidized Housing Coverage Ratio.

        Parameters:
        - gdf: GeoDataFrame with required columns.
        - output_file: Path to save the HTML map.

        Returns:
        - Folium Map object.
        """
        # Validate input
        required_cols = ['CensusTract', 'Total_LowIncome_Renter_HH', 'TotalUnits', 'Subsidized_Housing_Coverage_Ratio', 'geometry']
        if not all(col in gdf.columns for col in required_cols):
             missing = [col for col in required_cols if col not in gdf.columns]
             raise ValueError(f"Input 'gdf' is missing required columns: {missing}")
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("Input 'gdf' must be a GeoDataFrame.")

        # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
        if gdf.crs is None:
            print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
             gdf = gdf.to_crs(epsg=4326)

        # Calculate map center robustly
        if gdf.geometry.is_empty.any():
            print("Warning: GeoDataFrame contains empty geometries. Centroid calculation might be affected.")
            gdf_valid_geom = gdf[~gdf.geometry.is_empty]
            if gdf_valid_geom.empty:
                map_center = [39.8283, -98.5795]
                print("Warning: No valid geometries found. Using default map center.")
            else:
                centroid_projected = gdf_valid_geom.to_crs(epsg=5070).geometry.centroid
                centroid = centroid_projected.to_crs(epsg=4326)
                map_center = [centroid.y.mean(), centroid.x.mean()]
        else:
            centroid_projected = gdf.to_crs(epsg=5070).geometry.centroid
            centroid = centroid_projected.to_crs(epsg=4326)
            map_center = [centroid.y.mean(), centroid.x.mean()]

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
            include_lowest=True,
            right=True # Default, includes upper bound except for first bin
        )

        # Define color scheme (dark red to light orange/yellow - higher coverage is "better"/less red)
        color_map = {
            '[0–0.2]': '#a50026',  # Dark Red
            '(0.2–0.4]': '#d73027',  # Red
            '(0.4–0.6]': '#f46d43',  # Orange-Red
            '(0.6–0.8]': '#fdae61',  # Orange
            '>0.8': '#fee090'     # Light Yellow-Orange
        }
        # Alternative: Sequential Purple (Low = Dark, High = Light)
        # color_map = {
        #     '[0–0.2]': '#54278f',  # Dark Purple
        #     '(0.2–0.4]': '#756bb1',
        #     '(0.4–0.6]': '#9e9ac8',
        #     '(0.6–0.8]': '#cbc9e2',
        #     '>0.8': '#f2f0f7'     # Very Light Purple/Gray
        # }


        # Create a stepped colormap for legend
        max_ratio = gdf['Subsidized_Housing_Coverage_Ratio'].max() if not gdf.empty else 1.0
        colormap = StepColormap(
            colors=list(color_map.values()),
            index=[0, 0.2, 0.4, 0.6, 0.8, max_ratio], # Use actual max for upper bound
            vmin=0,
            vmax=max_ratio,
            caption='Subsidized Housing Coverage Ratio'
        )

        # Format Subsidized_Housing_Coverage_Ratio for tooltips
        gdf['CoverageRatioFormatted'] = gdf['Subsidized_Housing_Coverage_Ratio'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        gdf['Total_LowIncome_Renter_HH_Formatted'] = gdf['Total_LowIncome_Renter_HH'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        gdf['TotalUnits_Formatted'] = gdf['TotalUnits'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")


        # Define style function
        def style_function(feature):
            bin_label = feature['properties'].get('CoverageRatioBin')
            fill_color = color_map.get(bin_label, '#808080') # Default gray for missing/NaN
            if pd.isna(feature['properties'].get('Subsidized_Housing_Coverage_Ratio')):
                fill_color = '#808080'

            return {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 0.1,
                'fillOpacity': 0.7 if bin_label else 0.3
            }

        # Add GeoJson layer
        geojson_layer = folium.GeoJson(
            gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'Total_LowIncome_Renter_HH_Formatted', 'TotalUnits_Formatted', 'CoverageRatioFormatted'],
                aliases=['Census Tract:', 'Low-Income Renter HHs:', 'Subsidized Units:', 'Coverage Ratio:'],
                localize=True,
                sticky=True,
                labels=True,
                style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;")
            ),
            name='Subsidized Housing Coverage',
            highlight_function=lambda x: {'weight': 2, 'color': 'yellow', 'fillOpacity': 0.8}
        ).add_to(m)

        # Add colormap to map
        colormap.add_to(m)

        # Add footnote at the bottom using custom HTML
        footnote = """
        <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                    background-color: rgba(255, 255, 255, 0.85); padding: 8px; border: 1px solid grey; border-radius: 5px; z-index: 1000; max-width: 80%;">
            <p style="margin: 0; font-size: 13px; text-align: center;">
                <b>Coverage Ratio:</b> Subsidized Housing Units / Low-Income Renter Households (<$50k income) per tract.
                <b>Interpretation:</b> A low ratio (e.g., < 0.2, dark red) indicates few subsidized units relative to potential need. Gray areas indicate missing data.
            </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(footnote))

        # Add LayerControl
        folium.LayerControl().add_to(m)

        # Save the map
        m.save(output_file)
        print(f"Choropleth map saved to {output_file}")

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
    - Folium Map object.
    """
    # Input validation
    required_columns = ['CensusTract', 'expiration', 'AvgMonthsToExpire', 'TotalUnits', 'geometry']
    if not all(col in gdf.columns for col in required_columns):
        missing = [col for col in required_columns if col not in gdf.columns]
        raise ValueError(f"GeoDataFrame must contain columns: {required_columns}. Missing: {missing}")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input 'gdf' must be a GeoDataFrame.")

    # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
    if gdf.crs is None:
        print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Calculate map center robustly
    if gdf.geometry.is_empty.any():
        print("Warning: GeoDataFrame contains empty geometries. Centroid calculation might be affected.")
        gdf_valid_geom = gdf[~gdf.geometry.is_empty]
        if gdf_valid_geom.empty:
            map_center = [39.8283, -98.5795]
            print("Warning: No valid geometries found. Using default map center.")
        else:
            centroid_projected = gdf_valid_geom.to_crs(epsg=5070).geometry.centroid
            centroid = centroid_projected.to_crs(epsg=4326)
            map_center = [centroid.y.mean(), centroid.x.mean()]
    else:
        centroid_projected = gdf.to_crs(epsg=5070).geometry.centroid
        centroid = centroid_projected.to_crs(epsg=4326)
        map_center = [centroid.y.mean(), centroid.x.mean()]

    # Initialize the Folium map
    m = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")

    # Filter data for expiration categories and handle NaNs
    gdf_5_plus_yr = gdf[(gdf['expiration'] == 'expiration beyond 5 years') & gdf['AvgMonthsToExpire'].notna()].copy()
    gdf_5yr = gdf[(gdf['expiration'] == 'expiration within 5 years') & gdf['AvgMonthsToExpire'].notna()].copy()
    gdf_na = gdf[gdf['AvgMonthsToExpire'].isna()].copy() # Tracts with missing AvgMonthsToExpire

    # Create formatted columns for tooltips
    gdf_5_plus_yr['AvgMonthsToExpireFormatted'] = gdf_5_plus_yr['AvgMonthsToExpire'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    gdf_5yr['AvgMonthsToExpireFormatted'] = gdf_5yr['AvgMonthsToExpire'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    gdf['TotalUnitsFormatted'] = gdf['TotalUnits'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A") # Format all for tooltip


    # Create color scales
    # 5-plus-year expiration: Dark blue (low months, start at 60) to light blue (high months, capped at 600)
    min_val_10yr = 60 # Start scale at 60 months (5 years)
    max_val_10yr = 600 # Cap at 600 months (50 years)
    if not gdf_5_plus_yr.empty:
        # Adjust vmin if actual min is higher than 60, but ensure it's at least 60
        # vmin_10yr = max(gdf_5_plus_yr['AvgMonthsToExpire'].min(), min_val_10yr)
        vmin_10yr = min_val_10yr # Keep vmin fixed at 60 for consistency
        colormap_10yr = cm.LinearColormap(
            colors=['#08519c', '#bdd7e7'],  # Darker Blue to Lighter Blue
            # colors=['#4d004b', '#ccebc5'], # Alternative: Purple to Green
            vmin=vmin_10yr,
            vmax=max_val_10yr,
            caption='Avg Months to Expiration (> 5 Years, capped at 50 yrs)'
        )
        colormap_10yr.add_to(m) # Add legend

    # 5-year expiration: Dark red (low months, 0) to light red (high months, 60)
    min_val_5yr = 0
    max_val_5yr = 60
    if not gdf_5yr.empty:
        # vmax_5yr = max(gdf_5yr['AvgMonthsToExpire'].max(), max_val_5yr) # Adjust if max is > 60
        vmax_5yr = max_val_5yr # Keep vmax fixed at 60
        colormap_5yr = cm.LinearColormap(
            colors=['#a50f15', '#fcae91'],  # Darker Red to Lighter Red/Pink
            vmin=min_val_5yr,
            vmax=vmax_5yr,
            caption='Avg Months to Expiration (<= 5 Years)'
        )
        colormap_5yr.add_to(m) # Add legend


    # Add 5-plus-year expiration layer using GeoJson
    if not gdf_5_plus_yr.empty:
        style_function_10yr = lambda feature: {
            'fillColor': colormap_10yr(np.clip(feature['properties']['AvgMonthsToExpire'], min_val_10yr, max_val_10yr))
                         if pd.notna(feature['properties']['AvgMonthsToExpire']) else '#808080', # Gray if NaN (shouldn't happen due to filter)
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.7
        }

        folium.GeoJson(
            gdf_5_plus_yr,
            style_function=style_function_10yr,
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'AvgMonthsToExpireFormatted', 'TotalUnitsFormatted'],
                aliases=['Census Tract:', 'Avg Months to Expire:', 'Total Units in Tract:'],
                localize=True, sticky=True, labels=True,
                style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;")
            ),
            name='Expiration > 5 Years',
            highlight_function=lambda x: {'weight': 2, 'color': 'yellow', 'fillOpacity': 0.8},
            show=True # Show by default
        ).add_to(m)


    # Add 5-year expiration layer using GeoJson
    if not gdf_5yr.empty:
        style_function_5yr = lambda feature: {
            'fillColor': colormap_5yr(np.clip(feature['properties']['AvgMonthsToExpire'], min_val_5yr, max_val_5yr))
                         if pd.notna(feature['properties']['AvgMonthsToExpire']) else '#808080', # Gray if NaN (shouldn't happen)
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.7
        }

        folium.GeoJson(
            gdf_5yr,
            style_function=style_function_5yr,
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'AvgMonthsToExpireFormatted', 'TotalUnitsFormatted'],
                aliases=['Census Tract:', 'Avg Months to Expire:', 'Total Units in Tract:'],
                localize=True, sticky=True, labels=True,
                style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;")
            ),
            name='Expiration <= 5 Years',
            highlight_function=lambda x: {'weight': 2, 'color': 'yellow', 'fillOpacity': 0.8},
            show=True # Show by default
        ).add_to(m)

    # Add layer for tracts with missing expiration data
    if not gdf_na.empty:
         folium.GeoJson(
            gdf_na,
            style_function=lambda x: {'fillColor': '#808080', 'color':'black', 'weight': 0.1, 'fillOpacity': 0.5},
            tooltip=folium.GeoJsonTooltip(
                fields=['CensusTract', 'TotalUnitsFormatted'],
                aliases=['Census Tract:', 'Total Units in Tract:'],
                localize=True, sticky=True, labels=True,
                style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;")
            ),
            name='Missing Avg Expiration Data',
            highlight_function=lambda x: {'weight': 2, 'color': 'cyan', 'fillOpacity': 0.7},
            show=False # Hide by default
        ).add_to(m)


    # Add footnote using custom HTML div
    footnote_html = """
    <div style="position: fixed;
                bottom: 10px;
                left: 50%; transform: translateX(-50%);
                z-index: 9999;
                font-size: 13px;
                background-color: rgba(255, 255, 255, 0.85);
                padding: 8px; border: 1px solid grey; border-radius: 5px; max-width: 80%;">
        <b>Note:</b> Average months to expiration calculated per census tract.
        Expirations > 5 years are capped at 50 years (600 months) for visualization. Gray areas indicate missing average expiration data for the tract.
    </div>
    """
    m.get_root().html.add_child(folium.Element(footnote_html))

    # Add LayerControl to toggle between layers
    folium.LayerControl(collapsed=False).add_to(m)

    # Save the map
    m.save(output_path)
    print(f"Map saved to {output_path}")
    return m


def create_risk_category_barchart(df_grouped, save_path='data/charts/total_units_by_state_and_risk_category.png'):
    """
    Create a grouped bar chart showing TotalUnits by State and ConditionRiskCategory (side by side),
    excluding 'Unknown (No Score)', with space between states, x-axis labels rotated 90 degrees,
    and text labels only on the highest bar per state. Adjust layout to prevent legend cutoff. Upscaled text.

    Parameters:
    df_grouped (pd.DataFrame): DataFrame grouped by 'State' and 'ConditionRiskCategory' with 'TotalUnits' column.
    save_path (str): Path to save the plot image.
    """
    # Filter out 'Unknown (No Score)' category
    df_filtered = df_grouped[df_grouped['ConditionRiskCategory'] != 'Unknown (No Score)'].copy()

    # Define categories (excluding 'Unknown (No Score)') in desired risk order
    categories = [
        'Low Risk (Score >= 80)',
        'Moderate Risk (Score 60-79)',
        'Concern (H&S Non-Life-Threatening)',
        'High Risk (Score < 60)',
        'Urgent (H&S Life-Threatening)'
    ]

    # Define colors for each category (adjusting for better contrast/meaning if needed)
    color_schemes = {
        'Low Risk (Score >= 80)': '#90EE90',       # Light Green
        'Moderate Risk (Score 60-79)': '#FFFFE0',   # Light Yellow
        'Concern (H&S Non-Life-Threatening)': '#FFB6C1', # Light Red/Pink
        'High Risk (Score < 60)': '#FF7F50',     # Coral/Orange-Red (more visible than light purple)
        'Urgent (H&S Life-Threatening)': '#DC143C'  # Crimson Red (stronger than light blue)
    }
    # Original colors:
    # color_schemes = {
    #     'Low Risk (Score >= 80)': '#90EE90',  # Light green
    #     'Moderate Risk (Score 60-79)': '#FFFFE0',  # Light yellow
    #     'Concern (H&S Non-Life-Threatening)': '#FFB6C1',  # Light red
    #     'High Risk (Score < 60)': '#E6E6FA',  # Light purple
    #     'Urgent (H&S Life-Threatening)': '#ADD8E6'  # Light blue
    # }


    # Pivot the data for grouped bar chart
    pivot_df = df_filtered.pivot(index='State', columns='ConditionRiskCategory', values='TotalUnits').fillna(0)

    # Ensure all categories are present in the pivot table, even if they have no data
    for category in categories:
        if category not in pivot_df.columns:
            pivot_df[category] = 0

    # Reorder columns to match the categories list
    pivot_df = pivot_df[categories]

    # Sort states alphabetically for consistent plotting
    pivot_df = pivot_df.sort_index()

    # Create a figure and axis
    plt.figure(figsize=(26, 10)) # Wider figure for more states and larger text

    # Set the positions of the bars with spacing between states
    bar_width = 0.15  # Width of each bar
    states = pivot_df.index
    num_categories = len(categories)
    num_states = len(states)
    group_width = bar_width * num_categories  # Width of each group of bars
    spacing = 0.4  # Space between groups
    # Positions for the start of each state's group
    r = np.arange(num_states) * (group_width + spacing)

    # Plot bars for each category
    for idx, category in enumerate(categories):
        # Calculate the position for this category's bars within each group
        bar_positions = r + bar_width * idx
        heights = pivot_df[category]
        plt.bar(
            bar_positions,
            heights,
            width=bar_width,
            label=category,
            color=color_schemes[category],
            edgecolor='grey', # Add edge color for definition
            linewidth=0.5
        )

    # Add text labels only on the highest bar for each state
    for i, state in enumerate(states):
        # Get the heights of all bars for this state
        heights = pivot_df.loc[state].values
        max_height = max(heights) if len(heights) > 0 else 0 # Find the highest bar
        if max_height > 0:  # Only add text if there is a non-zero height
            max_idx = np.argmax(heights)  # Index of the highest bar
            bar_position = r[i] + bar_width * max_idx + bar_width / 2 # Center text on the highest bar
            plt.text(
                bar_position,  # x-position (center of bar)
                max_height,  # y-position (top of the highest bar)
                f'{int(max_height):,}',  # Format with commas
                ha='center',  # Center horizontally
                va='bottom',  # Align above the bar
                fontsize=10,  # Larger font size for labels
                color='black',
                rotation=0, # Keep text horizontal
                # Add background for readability if needed:
                # bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.2')
            )

    # Add labels and title with larger fonts
    plt.title('Total Units by State and Condition Risk Category (Excluding Unknown Score)', fontsize=20, pad=25)
    plt.xlabel('State', fontsize=16, labelpad=15)
    plt.ylabel('Total Units', fontsize=16, labelpad=15)

    # Set x-ticks to be in the middle of each group of bars and rotate 90 degrees
    plt.xticks([pos + group_width / 2 - bar_width / 2 for pos in r], states, rotation=90, ha='center', fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend with larger font size and adjusted position
    plt.legend(
        title='Condition Risk Category',
        bbox_to_anchor=(1.01, 1),  # Position legend outside plot area
        loc='upper left',
        fontsize=12,  # Larger font size for legend items
        title_fontsize=14  # Larger font size for legend title
    )

    # Use tight_layout first, then adjust if needed
    plt.tight_layout(rect=[0, 0, 0.98, 1]) # Adjust right margin slightly if legend still overlaps

    # Optionally add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # Save the plot with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


def create_risk_choropleth(gdf, output_path='data/risk_choropleth.html'):
    """
    Create a Folium choropleth map visualizing AvgMonthsToExpire colored by ConditionRiskCategory.

    Parameters:
    - gdf (GeoDataFrame): Contains CensusTract, ConditionRiskCategory, AvgMonthsToExpire, TotalUnits, geometry.
    - output_path (str): Path to save the HTML map.

    Returns:
    - Folium Map object.
    """
    # Input validation
    required_columns = ['CensusTract', 'ConditionRiskCategory', 'AvgMonthsToExpire', 'TotalUnits', 'geometry']
    if not all(col in gdf.columns for col in required_columns):
        missing = [col for col in required_columns if col not in gdf.columns]
        raise ValueError(f"GeoDataFrame must contain columns: {required_columns}. Missing: {missing}")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input 'gdf' must be a GeoDataFrame.")

    # Ensure the GeoDataFrame has a CRS; set to EPSG:4326 if none exists
    if gdf.crs is None:
        print("No CRS found. Setting CRS to EPSG:4326 (WGS84).")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Calculate map center robustly
    if gdf.geometry.is_empty.any():
        print("Warning: GeoDataFrame contains empty geometries. Centroid calculation might be affected.")
        gdf_valid_geom = gdf[~gdf.geometry.is_empty]
        if gdf_valid_geom.empty:
            map_center = [39.8283, -98.5795]
            print("Warning: No valid geometries found. Using default map center.")
        else:
            centroid_projected = gdf_valid_geom.to_crs(epsg=5070).geometry.centroid
            centroid = centroid_projected.to_crs(epsg=4326)
            map_center = [centroid.y.mean(), centroid.x.mean()]
    else:
        centroid_projected = gdf.to_crs(epsg=5070).geometry.centroid
        centroid = centroid_projected.to_crs(epsg=4326)
        map_center = [centroid.y.mean(), centroid.x.mean()]

    # Initialize the Folium map
    m = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")

    # Define categories (excluding 'Unknown (No Score)') - use the same order as barchart
    categories = [
        'Low Risk (Score >= 80)',
        'Moderate Risk (Score 60-79)',
        'Concern (H&S Non-Life-Threatening)',
        'High Risk (Score < 60)',
        'Urgent (H&S Life-Threatening)'
    ]

    # Define base colors for each category (can be single color or a gradient list)
    # Using single distinct colors for fill, as expiration is shown in tooltip, not color scale here
    # These match the revised barchart colors
    color_schemes = {
        'Low Risk (Score >= 80)': '#90EE90',       # Light Green
        'Moderate Risk (Score 60-79)': '#FFFFE0',   # Light Yellow
        'Concern (H&S Non-Life-Threatening)': '#FFB6C1', # Light Red/Pink
        'High Risk (Score < 60)': '#FF7F50',     # Coral/Orange-Red
        'Urgent (H&S Life-Threatening)': '#DC143C'  # Crimson Red
    }
    # If we want to color by AvgMonthsToExpire WITHIN each risk category:
    # expiration_colormap_colors = ['#edf8b1', '#7fcdbb', '#2c7fb8'] # Example Yellow to Blue for expiration time

    # Create formatted columns for tooltips
    gdf['AvgMonthsToExpireFormatted'] = gdf['AvgMonthsToExpire'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    gdf['TotalUnitsFormatted'] = gdf['TotalUnits'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")

    # Add a layer for each category using FeatureGroup for better layer control
    for category in categories:
        # Filter GeoDataFrame for the current category
        gdf_category = gdf[gdf['ConditionRiskCategory'] == category].copy()

        if not gdf_category.empty:
            # Create a FeatureGroup for this category
            fg = folium.FeatureGroup(name=category, show=True) # Show all categories by default

            # Get the color for this category
            category_color = color_schemes[category]

            # Define style function - simple fill based on category color
            style_function = lambda feature, color=category_color: {
                'fillColor': color if pd.notna(feature['properties']['AvgMonthsToExpire']) else '#808080', # Use category color, gray if no expiry data
                'color': 'black',
                'weight': 0.1,
                'fillOpacity': 0.7 if pd.notna(feature['properties']['AvgMonthsToExpire']) else 0.4
            }

            # Add GeoJson layer to the FeatureGroup
            folium.GeoJson(
                gdf_category,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['CensusTract', 'AvgMonthsToExpireFormatted', 'TotalUnitsFormatted', 'ConditionRiskCategory'],
                    aliases=['Census Tract:', 'Avg Months to Expire:', 'Total Units in Tract:', 'Risk Category:'],
                    localize=True, sticky=True, labels=True,
                    style=("background-color: white; color: black; font-family: sans-serif; font-size: 12px; padding: 5px;")
                ),
                 highlight_function=lambda x: {'weight': 2, 'color': 'yellow', 'fillOpacity': 0.9},
            ).add_to(fg)

            # Add the FeatureGroup to the map
            fg.add_to(m)

    # Add a separate legend for the risk categories (since fill color represents category)
    legend_html = '''
         <div style="position: fixed;
                     top: 10px; right: 10px; width: 180px; height: auto;
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color: white; padding: 5px; opacity: 0.9;">
         <b>Risk Category</b><br>
     '''
    for category in categories:
         legend_html += f'  <i class="fa fa-square" style="color:{color_schemes[category]}"></i>   {category}<br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))


    # Add LayerControl to toggle between categories (layers)
    folium.LayerControl(collapsed=False).add_to(m)

    # Add footnote at the bottom using custom HTML
    footnote = """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                background-color: rgba(255, 255, 255, 0.85); padding: 8px; border: 1px solid grey; border-radius: 5px; z-index: 1000; max-width: 80%;">
        <p style="margin: 0; font-size: 13px; text-align: center;">
            <b>NOTE:</b> Choropleth shows census tracts colored by the <b>Condition Risk Category</b> associated with properties within the tract.
            Hover over a tract for details including Average Months to Expiration. Gray areas indicate missing expiration data.
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(footnote))

    # Save the map
    m.save(output_path)
    print(f"Risk choropleth map saved to {output_path}")

    return m


def plot_subsidized_units_expiry_by_state(df, expiration_date, save_path='data/charts/subsidized_units_expiry_by_state.png'):
    """
    Plot subsidized units with expiring subsidies by state. Upscaled text and resolution.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'State' and 'TotalUnits' columns
    expiration_date (str): Date string for the footnote.
    save_path (str): Path to save the plot image.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Use a style with grid
    plt.figure(figsize=(18, 10), dpi=300) # Larger figure, higher DPI
    bar_width = 0.7 # Slightly wider bars if needed

    # Sort by TotalUnits for better visualization
    df_sorted = df.sort_values('TotalUnits', ascending=False)

    colors = sns.color_palette("Blues_d", n_colors=len(df_sorted['State'])) # Darker blues palette
    bars = plt.bar(df_sorted['State'], df_sorted['TotalUnits'],
                   color=colors, edgecolor='black', linewidth=0.7) # Darker edge

    # Labeling axes and title with larger fonts
    plt.xlabel('State', fontsize=16, labelpad=15, fontweight='bold')
    plt.ylabel('Total Units at Risk', fontsize=16, labelpad=15, fontweight='bold')
    plt.title(f'Subsidized Units with Subsidies Expiring Within 5 Years (by {expiration_date})',
              fontsize=20, pad=20, fontweight='bold')

    # Rotate x-tick labels to prevent overlap, larger font
    plt.xticks(rotation=90, ha='center', fontsize=12)
    plt.yticks(fontsize=12)

    # Add grid
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7) # Style already adds grid

    # Add text labels with better positioning and larger font
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height == 0: continue # Skip zero-height bars

        # Position text slightly above the bar
        text_position = height * 1.01 # Place text just above the bar top

        plt.text(bar.get_x() + bar.get_width() / 2,
                 text_position,
                 f'{int(height):,}',
                 ha='center',
                 va='bottom',  # Align the text bottom just above the bar
                 fontsize=10,  # Larger text size
                 color='black',
                 fontweight='medium' # Slightly bolder text
                 )

    # Adjust layout to ensure tight fit and avoid clipping
    footnote = f'Subsidies expiration: Within 5 years from {expiration_date}'
    # Add footnote using figtext
    plt.figtext(0.5, 0.01, footnote, wrap=True, horizontalalignment='center', fontsize=12, color='grey')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for footnote and title

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    plt.show()


def plot_rent_ratio_faceted_bubble_enhanced(df: pd.DataFrame, save_path='data/charts/rent_ratio_faceted_bubble.png') -> sns.axisgrid.FacetGrid:
    """
    Create a faceted scatter plot (bubble chart) of RentRatio by State and Expiration,
    with bubble size representing TotalUnits. Upscaled text and resolution.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'State', 'RentRatio', 'TotalUnits', 'expiration'.
    - save_path (str): Path to save the plot image.

    Returns:
    - sns.axisgrid.FacetGrid: The FacetGrid object.
    """
    # --- Create FacetGrid with custom figure size ---
    g = sns.relplot(
        data=df,
        x="State",
        y="RentRatio",
        size="TotalUnits", # Bubble size based on TotalUnits
        col="expiration", # Facet by expiration status
        col_order=['expiration within 5 years', 'expiration beyond 5 years'], # Specific order
        kind="scatter",
        col_wrap=1,       # Display facets vertically
        sizes=(150, 1200), # Larger range for bubble sizes
        alpha=0.75,       # Slightly more opaque bubbles
        height=8,         # Taller facets
        aspect=1.8,       # Wider aspect ratio for state labels
        legend=False,     # Custom legend/annotation might be better
        palette="viridis" # Color palette
    )

    # --- Add Text Labels Inside Bubbles ---
    for i, facet_name in enumerate(['expiration within 5 years', 'expiration beyond 5 years']):
        # Check if the facet exists (col_wrap=1 means axes are flat)
        if i < len(g.axes.flat):
            ax = g.axes.flat[i]
            data_subset = df[df['expiration'] == facet_name]

            # Sort subset by state to match x-axis order if necessary
            # data_subset = data_subset.set_index('State').loc[ax.get_xticklabels()].reset_index() # Requires careful handling if states differ

            # Add grid and background
            ax.set_facecolor('#f0f0f0') # Lighter gray background
            ax.grid(True, linestyle='--', alpha=0.6, color='white')

            # Iterate through data points for this facet
            for _, row in data_subset.iterrows():
                 # Try to find the correct x-position based on label matching (more robust)
                state_pos_list = [tick.get_text() for tick in ax.get_xticklabels()]
                try:
                    x_pos = state_pos_list.index(row['State'])
                except ValueError:
                    # If state not found on axis (shouldn't happen with default relplot), skip
                    continue

                # Add text label inside the bubble
                ax.text(
                    x=x_pos, # Use numerical index for x
                    y=row['RentRatio'],
                    s=f"{row['RentRatio']:.2f}",
                    fontsize=10,       # Larger font size for label
                    color='black',     # Contrast color
                    ha='center',
                    va='center',
                    # Add a subtle background for readability
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5, boxstyle='round,pad=0.2')
                )

    # --- Customize Axes ---
    g.set_xticklabels(rotation=90, fontsize=12) # Larger tick labels
    g.set_ylabels("Rent Ratio (Median Gross Rent / FMR)", fontsize=14) # Larger Y axis label
    g.set_xlabels("State", fontsize=14) # Add X axis label (relplot sometimes hides it)
    g.set_titles("Expiration Status: {col_name}", fontsize=16) # Larger facet titles

    # --- Adjust layout to make space for titles and footnote ---
    g.fig.subplots_adjust(top=0.90, bottom=0.15, hspace=0.3) # Adjust top, bottom, and horizontal space

    # --- Add Main Title ---
    g.fig.suptitle(
        "Rent Ratio vs. State, Faceted by Subsidy Expiration Status",
        fontsize=20,       # Larger main title
        weight='bold',
        y=0.98 # Position title slightly lower
    )

    # --- Add Footnote ---
    g.fig.text(
        x=0.5,
        y=0.02,
        s="Bubble size represents the average TotalUnits for the State and Expiration category. Rent Ratio = Median Gross Rent / Fair Market Rent (2BR).",
        ha='center', # Center align footnote
        va='bottom',
        fontsize=11,       # Larger footnote
        color='darkslategray'
    )

    # --- Save Final Plot ---
    g.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot as '{save_path}'")

    return g


def plot_subsidized_housing_boxplot(data_series,
                                    title="Subsidized Housing Coverage Ratio Distribution\n(Units per Low-Income Renter Household)",
                                    ylabel="Coverage Ratio", xlabel="All Census Tracts",
                                    save_path="data/charts/subsidized_housing_coverage_boxplot.png"):
        """
        Create a vertical box plot with y-axis capped at 0–0.6, outliers as small dots, IQR annotations,
        and upscaled text/resolution.

        Parameters:
        - data_series (pd.Series or array-like): Data to plot.
        - title (str): Plot title.
        - ylabel (str): Y-axis label.
        - xlabel (str): X-axis label.
        - save_path (str): Path to save the plot.

        Returns:
        - float: IQR value, or None if calculation fails.
        """
        try:
            # Convert input to pandas Series and drop NaNs/Infs
            data_series = pd.Series(data_series).replace([np.inf, -np.inf], np.nan).dropna()
            if len(data_series) == 0:
                print("Warning: No valid data points for box plot after cleaning.")
                return None

            # Calculate stats
            Q1 = data_series.quantile(0.25)
            Q3 = data_series.quantile(0.75)
            median = data_series.median()
            IQR = Q3 - Q1
            # Define whisker bounds (typically Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            lower_whisker = np.max([data_series.min(), Q1 - 1.5 * IQR])
            upper_whisker = np.min([data_series.max(), Q3 + 1.5 * IQR])

            # Identify points outside the 0-0.6 range for reporting
            clipped_points_upper = data_series[data_series > 0.6]
            clipped_points_lower = data_series[data_series < 0] # Though ratio shouldn't be negative

            print(f"Boxplot Stats: Q1={Q1:.3f}, Median={median:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
            print(f"Whisker Range (calculated): [{lower_whisker:.3f}, {upper_whisker:.3f}]")
            print(f"Points > 0.6: {len(clipped_points_upper)} ({len(clipped_points_upper)/len(data_series)*100:.2f}%)")
            if len(clipped_points_lower) > 0:
                 print(f"Warning: Points < 0 found: {len(clipped_points_lower)}")


            # Create the box plot
            plt.figure(figsize=(10, 8)) # Larger figure size
            box = plt.boxplot(data_series, vert=True, patch_artist=True, showfliers=True, # Show fliers
                        boxprops=dict(facecolor='lightblue', color='blue', linewidth=1.5),
                        whiskerprops=dict(color='blue', linestyle='--', linewidth=1.5),
                        capprops=dict(color='blue', linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2),
                        # Customize fliers: make them smaller and semi-transparent
                        flierprops=dict(marker='o', markersize=3, markerfacecolor='grey',
                                        markeredgecolor='none', alpha=0.3)) # Small, faint outliers

            # Add labels and title with larger fonts
            plt.title(title, fontsize=18, pad=15)
            plt.ylabel(ylabel, fontsize=16, labelpad=10)
            plt.xlabel(xlabel, fontsize=16, labelpad=10)
            plt.xticks([1], [xlabel], fontsize=14) # Keep label if desired
            plt.yticks(fontsize=12)


            # Set y-axis range to 0 to 0.6
            plt.ylim(-0.02, 0.62) # Slightly extend limits for visibility

            # Add grid
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Annotate IQR, Q1, Median, Q3 (adjust placement carefully if clipped)
            text_x_pos = 1.1 # Position for text annotations
            # Ensure annotation y-positions are within the visible range
            q3_label_y = min(Q3, 0.59)
            q1_label_y = max(Q1, 0.01)
            median_label_y = max(min(median, 0.59), 0.01)

            plt.text(text_x_pos, q3_label_y, f'Q3: {Q3:.3f}', fontsize=12, va='center', color='darkblue')
            plt.text(text_x_pos, q1_label_y, f'Q1: {Q1:.3f}', fontsize=12, va='center', color='darkblue')
            plt.text(text_x_pos, median_label_y, f'Median: {median:.3f}', fontsize=12, va='center', color='darkred', weight='bold')
            # Place IQR label between Q1 and Median for better spacing
            iqr_label_y = (q1_label_y + median_label_y) / 2
            plt.text(text_x_pos, iqr_label_y, f'IQR: {IQR:.3f}', fontsize=12, va='center', color='black')

            # Add annotation about clipping/outliers shown
            plt.text(0.95, -0.015, '*Y-axis limited to 0-0.6. Outliers shown as faint dots.',
                     fontsize=11, ha='center', va='top', color='grey')

            # Tight layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for title and footnote space

            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {os.path.abspath(save_path)}")

            # Show the plot
            plt.show()

            return IQR

        except Exception as e:
            print(f"Error creating box plot: {e}")
            return None


def plot_rent_ratio_faceted_bars(df, save_path='data/charts/rent_ratio_faceted_bars.png'):
            """
            Generate a faceted bar plot for RentRatio by state, with bar width encoding TotalUnits,
            excluding 'expired or unknown' expiration status. Upscaled text and resolution.

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

            # Filter out 'expired or unknown' and handle potential NaN/Inf in RentRatio/TotalUnits
            df_filtered = df[df['expiration'].isin(['expiration beyond 5 years', 'expiration within 5 years'])].copy()
            df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=['RentRatio', 'TotalUnits'])
            if df_filtered.empty:
                print("No valid data remains after filtering for faceted bar plot.")
                return

            # Sort states alphabetically for consistent y-axis order across facets
            sorted_states = sorted(df_filtered['State'].unique())
            df_filtered['State'] = pd.Categorical(df_filtered['State'], categories=sorted_states, ordered=True)

            # Normalize TotalUnits for bar height (used for thickness, range 0.2 to 0.8 looks good)
            min_units = df_filtered['TotalUnits'].min()
            max_units = df_filtered['TotalUnits'].max()
            if max_units == min_units: # Avoid division by zero if all units are the same
                df_filtered['BarHeight'] = 0.5 # Assign a medium height
            else:
                df_filtered['BarHeight'] = 0.2 + 0.6 * (df_filtered['TotalUnits'] - min_units) / (max_units - min_units)


            # Set Seaborn style
            sns.set_style("whitegrid")

            # Create faceted plot (horizontal bars)
            g = sns.FacetGrid(
                df_filtered,
                col='expiration',
                col_order=['expiration within 5 years', 'expiration beyond 5 years'],
                height=12,        # Taller figure for more states
                aspect=0.6,       # Adjust aspect ratio for horizontal bars
                sharey=True,      # Share Y axis (States)
                gridspec_kws={'wspace': 0.05} # Reduce space between facets
            )

            # Custom plotting function for horizontal bars with variable height (thickness)
            def custom_barplot_h(*args, **kwargs):
                data = kwargs.pop('data')
                ax = plt.gca()
                palette = {'expiration within 5 years': '#ff7f0e', 'expiration beyond 5 years': '#1f77b4'} # Orange/Blue
                color = palette.get(data['expiration'].iloc[0], 'grey') # Get color for the facet

                # Get state order from the shared Y axis
                state_order = [tick.get_text() for tick in ax.get_yticklabels()]
                state_map = {state: i for i, state in enumerate(state_order)}

                for _, row in data.iterrows():
                    state_y_pos = state_map.get(row['State'])
                    if state_y_pos is None: continue # Skip if state not found (shouldn't happen with sharey=True)

                    ax.barh(
                        y=state_y_pos,              # Use numeric position for y
                        width=row['RentRatio'],     # Length of bar is RentRatio
                        height=row['BarHeight'],    # Thickness of bar is normalized TotalUnits
                        color=color,
                        alpha=0.85,
                        edgecolor='black', linewidth=0.5
                    )
                    # Annotate RentRatio on bars (adjust x position for readability)
                    text_x = row['RentRatio'] * 0.98 # Place text just inside the end of the bar
                    ax.text(
                        text_x,
                        state_y_pos,
                        f'{row["RentRatio"]:.2f}',
                        ha='right',   # Align text to the right
                        va='center',
                        color='black', #'white' if row['RentRatio'] > (ax.get_xlim()[1] * 0.5) else 'black', # Adjust text color based on bar length?
                        fontsize=9,    # Slightly larger font size for annotation
                        weight='bold'
                    )
                # Set the yticks to use the state names
                ax.set_yticks(range(len(state_order)))
                ax.set_yticklabels(state_order)


            g.map_dataframe(custom_barplot_h)

            # Customize axes and titles with larger fonts
            g.set_titles('{col_name}', fontsize=16) # Larger facet titles
            g.set_xlabels('Rent Ratio (Median Gross Rent / FMR)', fontsize=14) # Larger X label
            g.set_ylabels('State', fontsize=14) # Larger Y label

            # Adjust axis limits and ticks
            max_rent_ratio = df_filtered['RentRatio'].max() * 1.05 # Add some padding
            for ax in g.axes.flat:
                ax.set_xlim(0, max_rent_ratio) # Start x-axis at 0
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=10) # Adjust y-tick label size if needed

            # Add overall title
            g.fig.suptitle('Rent Pressure by State and Subsidy Expiration', fontsize=20, y=1.02, weight='bold')

            # Add footnote about bar thickness
            g.fig.text(0.5, 0.01, 'Bar thickness represents relative Total Units within each state/expiration category.',
                       ha='center', va='bottom', fontsize=12, color='grey')

            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout rect for title/footnote

            # Save or display
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            # Close figure
            plt.close()


if __name__ == '__main__':

    # Create output directories if they don't exist
    os.makedirs('data/charts', exist_ok=True)
    os.makedirs('data', exist_ok=True) # Ensure base data directory exists too

    # NHPD stands for the National Housing Preservation Database. It helps identify properties at risk of losing their affordability due to expiring subsidies, helping prevent displacement and preserve affordable housing.
    print("Loading subsidized housing data...")
    try:
        subsidized_df = pd.read_excel('data/Active and Inconclusive Properties.xlsx')
        print(f"Loaded {len(subsidized_df)} records.")
    except FileNotFoundError:
        print("Error: 'data/Active and Inconclusive Properties.xlsx' not found.")
        print("Please ensure the NHPD data file is placed in the 'data' directory.")
        exit()

    # Idea: Predicting and Preventing the loss of Affordable Housing: Risk Assessment of Subsidized Properties
    # Objective: Develop a risk assessment framework to identify subsidized housing properties at greatest risk of losing their affordability status due to expiring subsidies.

    print("Processing subsidized housing data...")
    subsidized_df['EarliestEndDate'] = pd.to_datetime(subsidized_df['EarliestEndDate'], errors='coerce')
    subsidized_df['LatestEndDate'] = pd.to_datetime(subsidized_df['LatestEndDate'], errors='coerce')

    # consider only active property status (properties actively receiving subsidies)
    subsidized_df_v1  = subsidized_df[subsidized_df['PropertyStatus'] == 'Active'].copy() # Use .copy()
    print(f"Filtered to {len(subsidized_df_v1)} active properties.")

    # consider non-missing latest end date
    initial_len = len(subsidized_df_v1)
    subsidized_df_v1 = subsidized_df_v1[subsidized_df_v1['LatestEndDate'].notna()]
    print(f"Dropped {initial_len - len(subsidized_df_v1)} properties with missing 'LatestEndDate'. Remaining: {len(subsidized_df_v1)}")

    # Handle missing CensusTract
    initial_len = len(subsidized_df_v1)
    subsidized_df_v1 = subsidized_df_v1.dropna(subset=['CensusTract'])
    print(f"Dropped {initial_len - len(subsidized_df_v1)} properties with missing 'CensusTract'. Remaining: {len(subsidized_df_v1)}")

    # Format CensusTract
    subsidized_df_v1['CensusTract'] = subsidized_df_v1['CensusTract'].astype(int).astype(str).str.zfill(11)

    # --- Expiration Analysis ---
    print("Analyzing subsidy expirations...")
    current_date = datetime(2025, 4, 16) # fix current date to april 16, 2025
    print(f"Reference date for expiration analysis: {current_date.strftime('%Y-%m-%d')}")

    # Calculate MonthsToExpire (handle negative values for past expirations)
    subsidized_df_v1['MonthsToExpire'] = ((subsidized_df_v1['LatestEndDate'] - current_date) / np.timedelta64(1, 'D') / 30.44).round(1)

    # Time windows for expiration analysis
    five_years = current_date + pd.DateOffset(years=5)
    # ten_years = current_date + pd.DateOffset(years=10) # Not used in current logic, but defined

    # Categorize expiration status
    conditions = [
        (subsidized_df_v1['LatestEndDate'] >= current_date) & (subsidized_df_v1['LatestEndDate'] <= five_years),
        (subsidized_df_v1['LatestEndDate'] > five_years),
        (subsidized_df_v1['LatestEndDate'] < current_date) # Explicitly capture expired
    ]
    choices = ['expiration within 5 years', 'expiration beyond 5 years', 'expired prior to reference date']
    subsidized_df_v1['expiration'] = np.select(conditions, choices, default='unknown date logic') # Catch unexpected cases
    print("Expiration categories assigned:")
    print(subsidized_df_v1['expiration'].value_counts())


    # --- Units Expiring Within 5 Years ---
    expiring_5yr = subsidized_df_v1[subsidized_df_v1['expiration'] == 'expiration within 5 years'].copy()

    # Calculate total units at risk within 5 years
    units_5yr = expiring_5yr['TotalUnits'].sum()

    summary_5yr_by_state = expiring_5yr.groupby(['State']).agg(
        TotalUnits=('TotalUnits', 'sum'),
        PropertyCount=('NHPDPropertyID', 'count') # Count distinct properties
    ).reset_index().sort_values('TotalUnits', ascending=False)


    print("\n--- Subsidy Expiration Snapshot ---")
    print(f"Reference Date: {current_date.strftime('%B %d, %Y')}")
    print(f"Total units with subsidies expiring within 5 years (by {five_years.strftime('%B %d, %Y')}): {units_5yr:,}")

    print("\nTop 10 States - Units Expiring within 5 Years:")
    print(summary_5yr_by_state[['State', 'TotalUnits', 'PropertyCount']].head(10))

    # Plot units expiring by state
    print("\nGenerating plot for units expiring within 5 years by state...")
    plot_subsidized_units_expiry_by_state(summary_5yr_by_state, five_years.strftime('%B %d, %Y'))


    # --- Merge with Census Data ---
    print("\nLoading and merging with Census ACS data...")
    try:
        census_df = pd.read_parquet("data/acs5_2023_tract_v1_data_multi_state.parquet", engine="fastparquet")
        print(f"Loaded {len(census_df)} census tract records.")
    except FileNotFoundError:
        print("Error: 'data/acs5_2023_tract_v1_data_multi_state.parquet' not found.")
        print("Please ensure the Census data file is placed in the 'data' directory.")
        exit()

    # Ensure GEOID/CensusTract formats match
    census_df['GEOID'] = census_df['STATE']+census_df['COUNTY']+census_df['TRACT']
    census_df['CensusTract'] = census_df['GEOID'].astype(str).str.zfill(11)

    # Perform the merge
    df_subsidized_with_acs = pd.merge(subsidized_df_v1, census_df, on='CensusTract', how='left', suffixes=('', '_acs'))
    print(f"Merged subsidized data with ACS data. Resulting records: {len(df_subsidized_with_acs)}")

    # Handle geometry (assuming it's WKB in hex format)
    if 'geometry_acs' in df_subsidized_with_acs.columns: # Use geometry from ACS if available
        df_subsidized_with_acs['geometry'] = df_subsidized_with_acs['geometry_acs'].apply(lambda x: wkb.loads(bytes.fromhex(x), hex=False) if pd.notnull(x) and isinstance(x, str) else None)
    elif 'geometry' in df_subsidized_with_acs.columns and not isinstance(df_subsidized_with_acs['geometry'].iloc[0], gpd.array.GeometryDtype): # Check if already geometry
         # Assuming original geometry needs conversion
         df_subsidized_with_acs['geometry'] = df_subsidized_with_acs['geometry'].apply(lambda x: wkb.loads(x, hex=False) if pd.notnull(x) else None)

    # Save intermediate merged file
    # df_subsidized_with_acs.to_csv("data/active_subsidized_with_acs.csv", index=False) # Can be large, consider parquet
    # print("Saved merged data to data/active_subsidized_with_acs.csv")


    # --- Expiration Choropleth Map ---
    print("\nPreparing data for expiration choropleth map...")
    # Calculate average months to expire by census tract, weighted by units if desired, or simple mean
    # Simple mean approach:
    days_to_expire_by_tract_df = (
        df_subsidized_with_acs
        .groupby(['CensusTract', 'expiration']) # Group by tract AND expiration status
        .agg(
            AvgMonthsToExpire=('MonthsToExpire', 'mean'),
            TotalUnits=('TotalUnits', 'sum')
            )
        .reset_index()
    )
    # Filter out expired/unknown if necessary, or handle them in the map function
    # days_to_expire_by_tract_df = days_to_expire_by_tract_df[days_to_expire_by_tract_df['expiration'].isin(['expiration within 5 years', 'expiration beyond 5 years'])]

    # Add geometry
    tract_geometries = df_subsidized_with_acs[['CensusTract', 'geometry']].drop_duplicates(subset=['CensusTract']).set_index('CensusTract')
    days_to_expire_by_tract_df = days_to_expire_by_tract_df.merge(tract_geometries, on='CensusTract', how='left')

    # Convert to GeoDataFrame
    days_to_expire_by_tract_gdf = gpd.GeoDataFrame(days_to_expire_by_tract_df, geometry='geometry', crs="EPSG:4326") # Assume initial geometry is 4326
    days_to_expire_by_tract_gdf = days_to_expire_by_tract_gdf.dropna(subset=['geometry']) # Drop rows missing geometry for mapping
    print(f"Created GeoDataFrame for expiration map with {len(days_to_expire_by_tract_gdf)} tract/expiration combinations.")

    # Create Folium Map for Expiration
    print("Generating Months to Expiration choropleth map...")
    create_months_to_expiration_choropleth(days_to_expire_by_tract_gdf)


    # --- REAC Score / Condition Risk Analysis ---
    print("\n--- Analyzing Property Condition Risk (REAC Scores) ---")

    def parse_reac_score(score_str):
        """Parses REAC score string into numeric score and qualifier."""
        if pd.isna(score_str):
            return np.nan, None # Return None for qualifier
        score_str = str(score_str).strip()
        # Use regex to find numeric part and optional trailing letter (a, b, c, *, potentially others)
        match = re.match(r'^(\d+)([a-zA-Z*]?)$', score_str) # Allow letters or *
        if match:
            numeric_score = int(match.group(1))
            qualifier = match.group(2).lower() if match.group(2) else None
            # Ensure score is within valid range (0-100)
            if 0 <= numeric_score <= 100:
                return numeric_score, qualifier
            else: # Handle invalid numbers
                # print(f"Warning: Invalid REAC numeric score '{numeric_score}' found.")
                return np.nan, qualifier # Keep qualifier if present
        else: # Handle cases that don't match the pattern (e.g., just letters, multiple letters)
             # Check for health/safety flags without scores (e.g., 'b', 'c', 'b*', 'c*')
             qualifier_match = re.match(r'^([a-zA-Z*])$', score_str)
             if qualifier_match:
                  # print(f"Info: Found qualifier '{score_str}' without numeric score.")
                  return np.nan, score_str.lower() # No numeric score, just qualifier
             # print(f"Warning: Could not parse REAC score string: '{score_str}'")
             return np.nan, None


    df_property_risk_score = df_subsidized_with_acs.copy()
    print("Parsing REAC scores...")
    for i in [1, 2, 3]:
        score_col = f'ReacScore{i}'
        date_col = f'ReacScore{i}Date'
        if score_col in df_property_risk_score.columns:
             df_property_risk_score[[f'ReacScore{i}_Numeric', f'ReacScore{i}_Qualifier']] = \
                df_property_risk_score[score_col].apply(lambda x: pd.Series(parse_reac_score(x)))
             # Ensure date columns are datetime objects
             if date_col in df_property_risk_score.columns:
                df_property_risk_score[date_col] = pd.to_datetime(df_property_risk_score[date_col], errors='coerce')
        else:
             print(f"Warning: Column {score_col} not found.")
             df_property_risk_score[f'ReacScore{i}_Numeric'] = np.nan
             df_property_risk_score[f'ReacScore{i}_Qualifier'] = None


    # Determine the 'Most Recent' valid score/date/qualifier, prioritizing Score 1 -> 2 -> 3
    print("Identifying most recent REAC score for each property...")
    df_property_risk_score['MostRecentReacScore'] = np.nan
    df_property_risk_score['MostRecentReacQualifier'] = None
    df_property_risk_score['MostRecentReacDate'] = pd.NaT

    # Iterate backwards (3 -> 2 -> 1) to easily overwrite with more recent data
    for i in [3, 2, 1]:
        num_col = f'ReacScore{i}_Numeric'
        qual_col = f'ReacScore{i}_Qualifier'
        date_col = f'ReacScore{i}Date'

        if num_col in df_property_risk_score.columns:
            # Update only where the current score is valid (not NaN)
            # And optionally, only if the date is more recent (if dates are reliable)
            # Simple approach: prioritize based on column index (1 > 2 > 3)
            valid_score_mask = df_property_risk_score[num_col].notna() | df_property_risk_score[qual_col].notna() # Consider valid if score OR qualifier exists

            df_property_risk_score.loc[valid_score_mask, 'MostRecentReacScore'] = df_property_risk_score.loc[valid_score_mask, num_col]
            df_property_risk_score.loc[valid_score_mask, 'MostRecentReacQualifier'] = df_property_risk_score.loc[valid_score_mask, qual_col]
            if date_col in df_property_risk_score.columns:
                 df_property_risk_score.loc[valid_score_mask, 'MostRecentReacDate'] = df_property_risk_score.loc[valid_score_mask, date_col]


    # Define Risk Categories based on HUD guidance (prioritize H&S flags)
    def assign_risk(row):
        score = row['MostRecentReacScore']
        qualifier = row['MostRecentReacQualifier']
        # date = row['MostRecentReacDate'] # Date not currently used in category logic

        # Check if any score/qualifier exists
        if pd.isna(score) and qualifier is None:
            return "Unknown (No Score)"

        # Priority 1: Urgent H&S (Life-Threatening) - 'c' or 'c*'
        if qualifier and 'c' in qualifier:
            return "Urgent (H&S Life-Threatening)"

        # Priority 2: High Risk (Score < 60) - regardless of 'a' or 'b' flag
        if pd.notna(score) and score < 60:
             # Check if it wasn't already flagged as 'c'
             if not (qualifier and 'c' in qualifier):
                 return "High Risk (Score < 60)"

        # Priority 3: Concern (H&S Non-Life-Threatening) - 'b' or 'b*' flag
        if qualifier and 'b' in qualifier:
            # Check if not already flagged as 'c' or 'High Risk' score
            if not (qualifier and 'c' in qualifier) and not (pd.notna(score) and score < 60):
                 return "Concern (H&S Non-Life-Threatening)"

        # Priority 4: Moderate Risk (Score 60-79) - only if no 'b' or 'c' flag
        if pd.notna(score) and 60 <= score < 80:
            if not (qualifier and ('b' in qualifier or 'c' in qualifier)):
                 return "Moderate Risk (Score 60-79)"

        # Priority 5: Low Risk (Score >= 80) - only if no 'b' or 'c' flag
        if pd.notna(score) and score >= 80:
            if not (qualifier and ('b' in qualifier or 'c' in qualifier)):
                 return "Low Risk (Score >= 80)"

        # Catch-all for cases missed by above logic (e.g., only '*' qualifier)
        # Or if only a qualifier like 'a' exists without a score meeting other criteria
        if pd.isna(score) and qualifier is not None:
            return f"Unknown (Qualifier Only: {qualifier})" # Specific unknown category

        return "Unknown (Review Logic)" # Should not be reached ideally


    print("Assigning condition risk categories...")
    df_property_risk_score['ConditionRiskCategory'] = df_property_risk_score.apply(assign_risk, axis=1)

    print("\nCondition Risk Category Distribution:")
    print(df_property_risk_score['ConditionRiskCategory'].value_counts(dropna=False))

    # --- Risk Category Barchart ---
    print("\nGenerating risk category bar chart by state...")
    # Aggregate units by State and Risk Category
    df_property_risk_score_grouped = df_property_risk_score.groupby(['State', 'ConditionRiskCategory']).agg(
        TotalUnits=('TotalUnits','sum')
        ).reset_index()

    create_risk_category_barchart(df_property_risk_score_grouped)

    # --- Risk Choropleth Map ---
    print("\nPreparing data for risk choropleth map...")
    # Aggregate AvgMonthsToExpire and TotalUnits by CensusTract and Risk Category
    # Note: This assigns one risk category per tract based on properties within it.
    # A tract might have properties in multiple risk categories. This aggregates expiry time *within* each risk category present in the tract.
    property_condtion_risk_score_with_expiration_by_tract_df = (
        df_property_risk_score.groupby(['CensusTract','ConditionRiskCategory'])
        .agg(
            AvgMonthsToExpire=('MonthsToExpire', 'mean'), # Or weighted mean?
            TotalUnits=('TotalUnits', 'sum')
            )
    ).reset_index()

    # Optional: Filter out 'Unknown' categories for the map
    # property_condtion_risk_score_with_expiration_by_tract_df = property_condtion_risk_score_with_expiration_by_tract_df[
    #     ~property_condtion_risk_score_with_expiration_by_tract_df['ConditionRiskCategory'].str.startswith('Unknown')
    # ]

    # Add geometry
    property_condtion_risk_score_with_expiration_by_tract_df = property_condtion_risk_score_with_expiration_by_tract_df.merge(
        tract_geometries, on='CensusTract', how='left'
        )

    # Convert to GeoDataFrame
    property_condtion_risk_score_with_expiration_by_tract_gdf = gpd.GeoDataFrame(
        property_condtion_risk_score_with_expiration_by_tract_df, geometry='geometry', crs="EPSG:4326"
        )
    property_condtion_risk_score_with_expiration_by_tract_gdf = property_condtion_risk_score_with_expiration_by_tract_gdf.dropna(subset=['geometry'])
    print(f"Created GeoDataFrame for risk map with {len(property_condtion_risk_score_with_expiration_by_tract_gdf)} tract/risk combinations.")


    print("Generating Risk Category / Expiration choropleth map...")
    create_risk_choropleth(property_condtion_risk_score_with_expiration_by_tract_gdf)


    # --- Housing Coverage Analysis ---
    print("\n--- Analyzing Subsidized Housing Coverage ---")
    df_housing_gap = df_subsidized_with_acs.copy()

    # Define columns for low-income renter households (< $50k)
    low_income_cols = [
        'Renter_HH_Income_Less_5K', 'Renter_HH_Income_5K_10K', 'Renter_HH_Income_10K_15K',
        'Renter_HH_Income_15K_20K', 'Renter_HH_Income_20K_25K', 'Renter_HH_Income_25K_35K',
        'Renter_HH_Income_35K_50K'
    ]
    # Check if columns exist
    missing_income_cols = [col for col in low_income_cols if col not in df_housing_gap.columns]
    if missing_income_cols:
         print(f"Warning: Missing income columns in ACS data: {missing_income_cols}. Cannot calculate low-income HH sum.")
         # Skip coverage analysis if columns are missing
         can_run_coverage = False
    else:
         can_run_coverage = True
         # Select relevant columns and handle potential NaNs in income columns before summing
         df_housing_gap_v1 = df_housing_gap[['CensusTract','TotalUnits'] + low_income_cols].copy()
         # Fill NaNs with 0 for summation, assuming NaN means zero households in that bracket for a tract
         df_housing_gap_v1[low_income_cols] = df_housing_gap_v1[low_income_cols].fillna(0)

         # Calculate total low-income renter households
         df_housing_gap_v1['Total_LowIncome_Renter_HH'] = df_housing_gap_v1[low_income_cols].sum(axis=1)

         # Aggregate by Census Tract: Sum TotalUnits, take first Total_LowIncome_Renter_HH (should be unique per tract from ACS)
         subsidied_housing_coverage_df = df_housing_gap_v1.groupby('CensusTract').agg(
              Total_LowIncome_Renter_HH=('Total_LowIncome_Renter_HH', 'first'),
              TotalUnits=('TotalUnits', 'sum') # Sum units from all properties in the tract
              ).reset_index()

         # Define safe division function for ratio calculation
         def calculate_ratio(numerator, denominator):
             # Ensure inputs are numeric and denominator is valid
             num = pd.to_numeric(numerator, errors='coerce')
             den = pd.to_numeric(denominator, errors='coerce')
             if pd.isna(num) or pd.isna(den) or den <= 0:
                 return np.nan # Return NaN for invalid inputs or zero/negative denominator
             else:
                 return num / den

         print("Calculating Subsidized Housing Coverage Ratio...")
         subsidied_housing_coverage_df['Subsidized_Housing_Coverage_Ratio'] = subsidied_housing_coverage_df.apply(
              lambda row: calculate_ratio(row['TotalUnits'], row['Total_LowIncome_Renter_HH']),
              axis=1
         )
         print("Coverage Ratio statistics:")
         print(subsidied_housing_coverage_df['Subsidized_Housing_Coverage_Ratio'].describe())


         # --- Coverage Ratio Box Plot ---
         print("\nGenerating box plot for Subsidized Housing Coverage Ratio...")
         plot_subsidized_housing_boxplot(subsidied_housing_coverage_df['Subsidized_Housing_Coverage_Ratio'])


         # --- Coverage Choropleth Map ---
         print("\nPreparing data for housing coverage choropleth map...")
         # Add geometry
         subsidied_housing_coverage_df = subsidied_housing_coverage_df.merge(
              tract_geometries, on='CensusTract', how='left'
              )
         # Convert to GeoDataFrame
         subsidied_housing_coverage_gdf = gpd.GeoDataFrame(
              subsidied_housing_coverage_df, geometry='geometry', crs="EPSG:4326"
              )
         subsidied_housing_coverage_gdf = subsidied_housing_coverage_gdf.dropna(subset=['geometry'])
         print(f"Created GeoDataFrame for coverage map with {len(subsidied_housing_coverage_gdf)} tracts.")

         print("Generating Housing Coverage choropleth map...")
         create_housing_coverage_choropleth(subsidied_housing_coverage_gdf)

    if not can_run_coverage:
        print("\nSkipping Housing Coverage analysis due to missing income columns.")


    # --- Market Rent Pressure Analysis ---
    print("\n--- Analyzing Market Rent Pressure ---")
    market_rent_pressure_df = df_subsidized_with_acs.copy()

    # Required columns for rent ratio
    rent_ratio_cols = ['CensusTract', 'Median_Gross_Rent', 'FairMarketRent_2BR', 'expiration', 'TotalUnits', 'State', 'geometry']
    missing_rent_cols = [col for col in rent_ratio_cols if col not in market_rent_pressure_df.columns and col != 'geometry'] # Exclude geometry for now
    if 'geometry' not in market_rent_pressure_df.columns and 'geometry' not in tract_geometries.columns:
         missing_rent_cols.append('geometry')

    if missing_rent_cols:
        print(f"Warning: Missing columns required for Rent Ratio analysis: {missing_rent_cols}.")
        can_run_rent_ratio = False
    else:
        can_run_rent_ratio = True
        # Select relevant columns and drop rows with missing rent data needed for the ratio
        market_rent_pressure_df_v1 = market_rent_pressure_df[rent_ratio_cols].dropna(subset=['Median_Gross_Rent', 'FairMarketRent_2BR'])
        print(f"Filtered for Rent Ratio analysis: {len(market_rent_pressure_df_v1)} records with valid rent data.")

        # Aggregate by Census Tract: Average rents, sum units
        # Note: Averaging Median_Gross_Rent might not be statistically perfect, but simple for tract-level view
        market_rent_pressure_df_grp = market_rent_pressure_df_v1.groupby('CensusTract').agg(
            Median_Gross_Rent=('Median_Gross_Rent', 'mean'),
            FairMarketRent_2BR=('FairMarketRent_2BR', 'mean'),
            TotalUnits=('TotalUnits', 'sum'),
            # Keep State and expiration - might need mode() or first() if tract spans multiple
            State=('State', 'first'),
            expiration=('expiration', lambda x: x.mode()[0] if not x.mode().empty else 'unknown') # Most common expiration status in tract
        ).reset_index()

        print("Calculating Rent Ratio (Median Gross Rent / FMR 2BR)...")
        market_rent_pressure_df_grp['RentRatio'] = market_rent_pressure_df_grp.apply(
            lambda row: calculate_ratio(row['Median_Gross_Rent'], row['FairMarketRent_2BR']),
            axis=1
        )
        print("Rent Ratio statistics:")
        print(market_rent_pressure_df_grp['RentRatio'].describe())


        # --- Rent Ratio Box Plot ---
        print("\nGenerating box plot for Rent Ratio...")
        plot_rent_ratio_boxplot(market_rent_pressure_df_grp['RentRatio'].dropna(), # Ensure NaNs are dropped for boxplot
                                title="Distribution of Rent Ratio (Median Gross Rent / FMR 2BR)",
                                ylabel="Rent Ratio", figsize=(10, 8))


        # --- Rent Ratio Choropleth Map ---
        print("\nPreparing data for Rent Ratio choropleth map...")
        # Merge geometry back
        market_rent_pressure_df_grp = market_rent_pressure_df_grp.merge(
            tract_geometries, on='CensusTract', how='left'
            )
        # Convert to GeoDataFrame
        market_rent_pressure_df_grp_gdf = gpd.GeoDataFrame(
            market_rent_pressure_df_grp, geometry='geometry', crs="EPSG:4326"
            )
        market_rent_pressure_df_grp_gdf = market_rent_pressure_df_grp_gdf.dropna(subset=['geometry', 'RentRatio']) # Drop if geometry or ratio is missing
        print(f"Created GeoDataFrame for Rent Ratio map with {len(market_rent_pressure_df_grp_gdf)} tracts.")

        print("Generating Rent Ratio choropleth map...")
        create_rent_ratio_choropleth(market_rent_pressure_df_grp_gdf)


        # --- Rent Ratio Faceted Plots ---
        print("\nPreparing data for faceted Rent Ratio plots by State and Expiration...")
        # Aggregate RentRatio and TotalUnits by State and Expiration status
        # Use the tract-level aggregated data for consistency
        rent_ratio_by_state_expiration_df = market_rent_pressure_df_grp.groupby(["State", "expiration"]).agg(
            RentRatio=('RentRatio', 'mean'), # Average tract-level RentRatio per state/expiration
            TotalUnits=('TotalUnits', 'sum')   # Sum of units per state/expiration
        ).reset_index()

        # Filter out non-relevant expiration statuses
        rent_ratio_by_state_expiration_df = rent_ratio_by_state_expiration_df[
             rent_ratio_by_state_expiration_df['expiration'].isin(['expiration within 5 years', 'expiration beyond 5 years'])
             ].copy()
        print(f"Aggregated Rent Ratio data for {len(rent_ratio_by_state_expiration_df)} state/expiration combinations.")

        # Generate Faceted Bar Plot
        print("Generating faceted bar plot for Rent Ratio...")
        plot_rent_ratio_faceted_bars(rent_ratio_by_state_expiration_df)

        # Generate Faceted Bubble Plot
        print("Generating faceted bubble plot for Rent Ratio...")
        # Calculate AVERAGE units per state/expiration for bubble size (as per original function description)
        rent_ratio_bubble_df = market_rent_pressure_df_grp.groupby(["State", "expiration"]).agg(
            RentRatio=('RentRatio', 'mean'),
            TotalUnits=('TotalUnits', 'mean') # Use mean for bubble size
        ).reset_index()
        rent_ratio_bubble_df = rent_ratio_bubble_df[
             rent_ratio_bubble_df['expiration'].isin(['expiration within 5 years', 'expiration beyond 5 years'])
             ].copy()

        plot_grid_final = plot_rent_ratio_faceted_bubble_enhanced(rent_ratio_bubble_df)
        # plt.show() # Show the bubble plot - already shown inside function? Check function.

    if not can_run_rent_ratio:
        print("\nSkipping Market Rent Pressure analysis due to missing columns.")


    print("\n--- Analysis Complete ---")