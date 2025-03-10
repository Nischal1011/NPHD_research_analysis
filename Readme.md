# National Housing Preservation Analysis

This project aims to provide a risk assessment framework to identify subsidized housing properties that are at risk of losing their affordability status due to expiring subsidies. It leverages data from the National Housing Preservation Database (NHPD) and various other data sources to make informed decisions on housing management and preservation.

**Subsidy Expiration**

https://github.com/user-attachments/assets/96039cf8-8050-431b-85c1-d9bd3e332030

**Property Risk**

<img width="1372" alt="image" src="https://github.com/user-attachments/assets/c9fd7248-7686-4ff7-a0df-d9eb3143e42a" />



**Housing Gap by segments**
![image](https://github.com/user-attachments/assets/887c08fc-7d90-41f7-bd53-0cb30ec6c22a)



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project requires Python 3.6 or higher. To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **API Keys**: Set up your API keys in a `.env` file to access external data sources such as the Census API and OpenAI (for potential analysis extensions).
   ```plaintext
   census_key='YOUR_CENSUS_API_KEY'
   ```

2. **Data Preparation**: Ensure that the dataset `Active and Inconclusive Properties.xlsx` which can be requested from NHPD website and is available in a `data/` directory.

3. **Run the Analysis**: Execute the data analysis scripts for various functionalities.
   - `nhpd.py`: Analyze properties at risk due to expiring subsidies.
   - `stats_modelling_nhpd.py`: Perform statistical modeling on the data including risk prediction.
   - `target_variable_analysis.py`: Analyze target variables and composite risk metrics.

4. **Additional Scripts**: Use `Wrapper/census.py` to fetch additional census tract data if needed.

## Features

- **Risk Assessment**: Identify properties at risk of losing their affordable status.
- **Data Visualization**: Generate visual summaries of the property condition and expiring subsidies.
- **Economic Alignment Analysis**: Evaluate housing gaps in relation to economic needs.
- **Statistical Modeling**: Predict property conditions using advanced statistical models.
- **Mapping**: Visual representation of properties at risk with Folium maps.

## Data Sources

- **National Housing Preservation Database**: Provides the main dataset with details about properties including subsidies and expiration dates.
- **U.S. Census Bureau**: Provides additional demographic and economic data for analysis.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](License.md) file for more details.
