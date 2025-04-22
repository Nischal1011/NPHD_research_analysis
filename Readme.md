# National Housing Preservation Analysis Framework

This project provides a multi-dimensional risk assessment framework for federally subsidized housing properties. It aims to identify properties facing various preservation risks, including expiring subsidies, potential physical deterioration, challenging market conditions, and inadequate supply relative to local need, supporting informed preservation strategies, particularly for agencies like HUD.

It leverages data from the National Housing Preservation Database (NHPD) and the American Community Survey (ACS) to conduct descriptive spatial analyses and predictive modeling.

## Key Risk Dimensions Analyzed

**1. Subsidy Expiration Risk:** Identifies properties with affordability contracts expiring in the near term.

   <img width="1696" alt="expiration_risk_within_5_yrs_by_state" src="https://github.com/user-attachments/assets/a7e6f63a-7eea-43b1-aa8c-7c575a09e590" />


**2. Property Condition Risk:** Assesses physical condition using available REAC scores, categorizing properties into risk levels (Urgent, High Risk, Concern, Moderate, Low). Property age is also considered as a key indicator, especially given REAC data availability.

   ![condition_risk_map](https://github.com/user-attachments/assets/26bd3534-f239-40ff-808e-f07d109f1c23)

   

**3. Subsidized Housing Coverage Ratio:** Measures the ratio of subsidized units to low-income renter households per census tract, highlighting areas of potential undersupply.

![subsidized_housing_risk_ratio_map](https://github.com/user-attachments/assets/e5c51104-5fe3-4da4-89b7-2f412cc48c4f)


**4. Market Rent Pressure (Rent Ratio):** Compares local Median Gross Rent (ACS) to Fair Market Rent (FMR) to identify areas where market rates may significantly exceed affordability benchmarks, creating potential opt-out pressure.

![rent_ratio_map](https://github.com/user-attachments/assets/6824f7ff-2cda-426f-b926-dc39d843eb3f)


**5. Location in High Stress Environments (Predictive Modeling):** Uses an XGBoost model with SHAP explainability to predict the likelihood of a property being located in a census tract defined by high rent burden and low relative income. Key drivers identified include poverty, disability rates, property age, and FMR.

<img width="1289" alt="shap_result" src="https://github.com/user-attachments/assets/e4b81943-d935-4570-9310-acc6581e716e" />

<img width="1594" alt="roc_curve_model" src="https://github.com/user-attachments/assets/4b596fcf-fe0c-4bf6-bcb0-33962ef1e731" />


---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project requires Python 3.9 or higher. To set up your environment and install the necessary packages:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Nischal1011/NPHD_research_analysis.git
    cd NPHD_research_analysis
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **API Keys**: If using the Census data fetching capabilities, set up your API key in a `.env` file in the project root:
    ```plaintext
    census_key='YOUR_CENSUS_API_KEY'
    ```
2.  **Data**: Ensure the primary dataset `Active and Inconclusive Properties.xlsx` (requested from the NHPD website) is placed in the `data/` directory.
3.  **Run Analysis**: Explore the Jupyter notebooks or Python scripts within the repository (e.g., `notebooks/`, `src/`) to execute specific parts of the analysis pipeline, including data preprocessing, risk metric calculation, model training, and visualization generation. Refer to individual script/notebook documentation for detailed instructions.
    *   Key scripts might include `nhpd.py` (initial processing/expiration), `stats_modelling_nhpd.py` (modeling), and various analysis-specific scripts/notebooks.

## Features

-   **Multi-Dimensional Risk Assessment**: Analyzes subsidy expiration, property condition proxies, market rent pressure, and housing coverage gaps.
-   **Predictive Modeling**: Identifies properties in 'High Stress Environments' using XGBoost.
-   **Model Interpretability**: Leverages SHAP values to understand key drivers of neighborhood stress indicators.
-   **Geospatial Visualization**: Creates maps illustrating the geographic distribution of various risk factors using libraries like GeoPandas and Folium/Matplotlib.
-   **Data Integration**: Combines property-level data from NHPD with tract-level socioeconomic context from ACS.
-   **Modular Analysis Scripts**: Organizes analysis into potentially reusable scripts and notebooks.

## Data Sources

-   **National Housing Preservation Database (NHPD)**: Primary source for subsidized housing property details, subsidy information, and some property characteristics. Sourced from NLIHC and PAHRC.
-   **U.S. Census Bureau American Community Survey (ACS)**: Provides tract-level demographic, socioeconomic, and housing data (e.g., income, poverty, rent burden, disability). Utilizes 5-Year Estimates (e.g., 2019-2023).

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](License.md) file for more details.
