import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import os

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Load the dataset
df = pd.read_csv('data/df_with_target.csv', low_memory=False)


# Assuming df is your DataFrame
df[['ReacScore1', 'ReacScore2', 'ReacScore3']] = df[['ReacScore1', 'ReacScore2', 'ReacScore3']].replace({'[a-zA-Z]': ''}, regex=True).apply(pd.to_numeric, errors='coerce')

# Define relevant features
relevant_features = [
    'TotalUnits', 'StudioOneBedroomUnits', 'TwoBedroomUnits', 'ThreePlusBedroomUnits',
    'OccupancyRate', 'AverageMonthsOfTenancy', 'PercentofELIHouseholds', 'FairMarketRent_2BR',
    'ReacScore1', 'ReacScore2', 'ReacScore3',
    'ActiveSubsidies', 'TotalInconclusiveSubsidies', 'TotalInactiveSubsidies',
    'NumberActiveSection8', 'NumberInactiveSection8', 'NumberActiveSection202', 'NumberInactiveSection202',
    'NumberActiveSection236', 'NumberInactiveSection236', 'NumberActiveHUDInsured', 'NumberInactiveHud',
    'NumberActiveLihtc', 'NumberInactiveLihtc', 'NumberActiveSection515', 'NumberInactiveSection515',
    'NumberActiveSection538', 'NumberInactiveSection538', 'NumberActiveHome', 'NumberInactiveHome',
    'NumberActivePublicHousing', 'NumberInactivePublicHousing', 'NumberActiveState', 'NumberInactiveState',
    'NumberActivePBV', 'NumberInactivePBV', 'NumberActiveMR', 'NumberInactiveMR',
    'NumberActiveNHTF', 'NumberInactiveNHTF',
    'S8_1_AssistedUnits', 'S8_2_AssistedUnits', 'S202_1_AssistedUnits', 'S202_2_AssistedUnits',
    'S236_1_AssistedUnits', 'S236_2_AssistedUnits', 'FHA_1_AssistedUnits', 'FHA_2_AssistedUnits',
    'LIHTC_1_AssistedUnits', 'LIHTC_2_AssistedUnits', 'RHS515_1_AssistedUnits', 'RHS515_2_AssistedUnits',
    'RHS538_1_AssistedUnits', 'RHS538_2_AssistedUnits', 'HOME_1_AssistedUnits', 'HOME_2_AssistedUnits',
    'PH_1_AssistedUnits', 'PH_2_AssistedUnits', 'State_1_AssistedUnits', 'State_2_AssistedUnits',
    'Pbv_1_AssistedUnits', 'Pbv_2_AssistedUnits', 'Mr_1_AssistedUnits', 'Mr_2_AssistedUnits', 'NHTF_2_AssistedUnits',
    'Census_Population_Total', 'Census_White_Population', 'Census_Black_Population', 'Census_Asian_Population',
    'Census_Hispanic_Population', 'Census_Median_Home_Value', 'Census_Educational_Attainment_College_Degree',
    'Census_Labor_Force_Participation', 'Census_Year_Built_1950_1979', 'Census_Year_Built_1980_1999',
    'Census_Year_Built_2000_and_Above', 'Census_Year_Built_1970_1979', 'Census_Year_Built_2010_Above',
    'Census_Year_Built_1960_1969', 'Census_Year_Built_1940_1949', 'Census_Year_Built_1950_1959',
    'Census_Age_25_44', 'Census_Age_45_64', 'Census_Age_65_74', 'Census_Age_75_and_older',
    'Census_Migration_In', 'Census_Migration_Out',
    'Risk_Combined'
]

# Filter dataframe
df = df[relevant_features]

# Feature engineering
df['S8_AssistedUnits'] = df[['S8_1_AssistedUnits', 'S8_2_AssistedUnits']].max(axis=1)
df['S8_AssistedRatio'] = df['S8_AssistedUnits'] / df['TotalUnits'].replace(0, np.nan)
df['LIHTC_AssistedUnits'] = df[['LIHTC_1_AssistedUnits', 'LIHTC_2_AssistedUnits']].max(axis=1)
df['LIHTC_AssistedRatio'] = df['LIHTC_AssistedUnits'] / df['TotalUnits'].replace(0, np.nan)
subsidy_cols = [col for col in df.columns if col.startswith('NumberActive')]
df['NumActiveSubsidies'] = (df[subsidy_cols] > 0).sum(axis=1)

# Handle missing values
subsidy_features = [col for col in df.columns if 'NumberActive' in col or 'NumberInactive' in col or 'AssistedUnits' in col or 'AssistedRatio' in col]
df[subsidy_features] = df[subsidy_features].fillna(0)
other_num_features = [col for col in df.select_dtypes(include=[np.number]).columns if col not in subsidy_features and col != 'Risk_Combined']
imputer = SimpleImputer(strategy='median')
df[other_num_features] = imputer.fit_transform(df[other_num_features])

# Separate features and target
X = df.drop(columns=['Risk_Combined'])
y = df['Risk_Combined']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weight for XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=scale_pos_weight)
xgb_classifier.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_proba = xgb_classifier.predict_proba(X_test_scaled)[:, 1]

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"\nXGBoost AUC-ROC: {auc_roc:.3f}")

# Cross-validation
cv_scores = cross_val_score(xgb_classifier, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"5-Fold CV AUC-ROC: {cv_scores.mean():.3f} (± {cv_scores.std() * 2:.3f})")
print("CV Scores:", cv_scores)

# Feature importance (Gain)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nTop 20 Feature Importance (Gain):")
print(feature_importance.head(20))

# SHAP analysis
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot (shows feature impact and direction)
print("\nGenerating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", max_display=20)

# Detailed SHAP summary plot (shows distribution of impacts)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, max_display=20)