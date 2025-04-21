import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('data/active_subsidized_with_acs.csv', low_memory=False)
    print(f"Data loaded with shape: {df.shape}")

    # --- Step 1: Define the Target Variable (IsInHighStressTract) ---
    print("Defining target variable 'IsInHighStressTract'...")
    df['Total_Renter_Occupied_Units_For_Rent_Burden'] = df['Total_Renter_Occupied_Units_For_Rent_Burden'].replace(0, np.nan)
    df['Percent_Cost_Burdened'] = (
        df['Rent_30_35_Percent_Income'].fillna(0) +
        df['Rent_35_40_Percent_Income'].fillna(0) +
        df['Rent_40_50_Percent_Income'].fillna(0) +
        df['Rent_50_Percent_Or_More_Income'].fillna(0)
    ) / df['Total_Renter_Occupied_Units_For_Rent_Burden'] * 100

    state_col = 'State' if 'State' in df.columns else 'STATE'
    state_income_quartiles = df.groupby(state_col)['Median_Household_Income'].quantile(0.25).reset_index()
    state_income_quartiles.columns = [state_col, 'Income_25th_Percentile']
    df = df.merge(state_income_quartiles, on=state_col, how='left')

    # Impute missing values for target definition using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df[['Percent_Cost_Burdened', 'Median_Household_Income', 'Income_25th_Percentile']] = imputer.fit_transform(
        df[['Percent_Cost_Burdened', 'Median_Household_Income', 'Income_25th_Percentile']]
    )

    df['IsInHighStressTract'] = (
        (df['Percent_Cost_Burdened'] > 50) &
        (df['Median_Household_Income'] < df['Income_25th_Percentile'])
    ).astype(int)

    df = df.drop(['Percent_Cost_Burdened', 'Income_25th_Percentile'], axis=1, errors='ignore')
    print("\nTarget Distribution:")
    print(df['IsInHighStressTract'].value_counts(normalize=True))
    if df['IsInHighStressTract'].nunique() < 2:
        raise ValueError("Target variable 'IsInHighStressTract' has only one class.")

    # --- Step 2: Enhanced Feature Engineering ---
    print("\nEngineering property-level and demographic features...")
    
    # Property Age (using EarliestConstructionDate)
    df['EarliestConstructionDate'] = pd.to_datetime(df['EarliestConstructionDate'], errors='coerce')
    df['PropertyAge'] = (pd.to_datetime('2025-04-16') - df['EarliestConstructionDate']).dt.days / 365.25
    df['PropertyAge'] = df['PropertyAge'].clip(lower=0)  # Avoid negative ages

    # Subsidy Density
    subsidy_cols = ['NumberActiveSection8', 'NumberActiveLihtc', 'NumberActiveSection202', 
                    'NumberActiveSection515', 'NumberActivePublicHousing']
    df['TotalActiveSubsidies'] = df[subsidy_cols].sum(axis=1)
    df['SubsidyDensity'] = df['TotalActiveSubsidies'] / df['TotalUnits'].replace(0, 1)

    # Unit Distribution Features
    df['PercentStudioOneBedroom'] = df['StudioOneBedroomUnits'].fillna(0) / df['TotalUnits'].replace(0, 1)
    df['PercentTwoBedroom'] = df['TwoBedroomUnits'].fillna(0) / df['TotalUnits'].replace(0, 1)
    df['PercentThreePlusBedroom'] = df['ThreePlusBedroomUnits'].fillna(0) / df['TotalUnits'].replace(0, 1)

    # Subsidy Status
    df['MonthsToExpire'] = pd.to_numeric(df['MonthsToExpire'], errors='coerce')
    df['Has_ExpiredSubsidy'] = (df['MonthsToExpire'] < 0).astype(int)
    for col in subsidy_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[f'Has_{col.split("NumberActive")[1]}'] = (df[col] > 0).astype(int)

    # Demographic Features
    df['Poverty_Rate'] = df['Population_Below_Poverty'] / df['Total_Population_Poverty_Status'].replace(0, 1)
    df['Elderly_Population_Ratio'] = (
        df[['Male_65_66_Years', 'Male_67_69_Years', 'Male_70_74_Years', 'Male_75_Years_And_Over',
            'Female_65_66_Years', 'Female_67_69_Years', 'Female_70_74_Years', 'Female_75_Years_And_Over']].sum(axis=1)
    ) / df['Total_Population'].replace(0, 1)
    df['Disability_Rate'] = (
        df[['Male_Under_5_With_Disability', 'Male_5_17_With_Disability', 'Male_18_34_With_Disability',
            'Male_35_64_With_Disability', 'Male_65_74_With_Disability', 'Male_75_Over_With_Disability',
            'Female_Under_5_With_Disability', 'Female_5_17_With_Disability', 'Female_18_34_With_Disability',
            'Female_35_64_With_Disability', 'Female_65_74_With_Disability', 'Female_75_Over_With_Disability']].sum(axis=1)
    ) / df['Total_Population'].replace(0, 1)

    # --- Step 3: Clean Categorical Features ---
    print("\nCleaning categorical features...")
    if 'CBSAType' in df.columns:
        df['CBSAType'] = df['CBSAType'].replace({
            'Metropolitan Statistical Area': 'Metro', 'Micropolitan Statistical Area': 'Micro',
            'Metropolitan': 'Metro', 'Micropolitan': 'Micro'
        }).fillna('Unknown')
        df['IsUrban'] = (df['CBSAType'] == 'Metro').astype(int)
    else:
        df['CBSAType'] = 'Unknown'
        df['IsUrban'] = 0

    for col in ['OwnerType', 'TargetTenantType']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
            df[col] = df[col].replace({
                'NON PROFIT': 'NonProfit', 'NON-PROFIT': 'NonProfit', 'FOR PROFIT': 'ForProfit',
                'PROFIT MOTIVATED': 'ForProfit', 'LIMITED PROFIT': 'LimitedProfit', 'LIMITED DIVIDEND': 'LimitedProfit',
                'PUBLIC HOUSING AGENCY': 'PublicAgency', 'PUBLIC AGENCY': 'PublicAgency',
                'PUBLIC ENTITY': 'PublicAgency', 'PUBLIC HOUSING AUTHORITY': 'PublicAgency',
                'FAMILY': 'Family', 'ELDERLY OR DISABLED': 'Elderly/Disabled', 'ELDERLY': 'Elderly/Disabled',
                'DISABLED': 'Elderly/Disabled', 'MIXED': 'Mixed', 'MIXED INCOME': 'Mixed'
            })
            counts = df[col].value_counts(normalize=True)
            common = counts[counts >= 0.01].index
            df[col] = df[col].apply(lambda x: x if x in common else 'Other').fillna('Unknown')
        else:
            df[col] = 'Unknown'

    # --- Step 4: Select Features and Define Preprocessing ---
    print("\nSelecting features and defining preprocessing...")
    numerical_features = [
        'TotalUnits', 'StudioOneBedroomUnits', 'TwoBedroomUnits', 'ThreePlusBedroomUnits',
        'PercentStudioOneBedroom', 'PercentTwoBedroom', 'PercentThreePlusBedroom',
        'S8_1_AssistedUnits', 'LIHTC_1_AssistedUnits', 'MonthsToExpire', 'FairMarketRent_2BR',
        'PropertyAge', 'SubsidyDensity', 'Poverty_Rate', 'Elderly_Population_Ratio', 'Disability_Rate'
    ]
    categorical_features = ['OwnerType', 'TargetTenantType', 'CBSAType']
    binary_features = [
        'Has_Section8', 'Has_LIHTC', 'Has_Section202', 'Has_Section515', 'Has_PublicHousing',
        'Has_ExpiredSubsidy', 'IsUrban'
    ]

    numerical_features = [f for f in numerical_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    binary_features = [f for f in binary_features if f in df.columns]
    all_features = numerical_features + categorical_features + binary_features
    print(f"Selected features: {all_features}")

    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features),
            ('bin', binary_pipeline, binary_features)
        ],
        remainder='drop'
    )

    y = df['IsInHighStressTract']
    X = df[all_features]

    # --- Step 5: Train-Test Split ---
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- Step 6: Preprocessing and SMOTE ---
    print("\nApplying preprocessing and SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    # Apply SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"Resampled train shape: {X_train_resampled.shape}")

    # --- Step 7: Model Training and Tuning (XGBoost and LightGBM) ---
    print("\nTraining models...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc', random_state=42
    )
    xgb_param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'scale_pos_weight': [scale_pos_weight, 1]
    }

    # LightGBM Model
    lgb_model = lgb.LGBMClassifier(
        objective='binary', metric='auc', random_state=42, verbosity=-1
    )
    lgb_param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'scale_pos_weight': [scale_pos_weight, 1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # XGBoost GridSearch
    xgb_grid = GridSearchCV(
        xgb_model, xgb_param_grid, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1
    )
    xgb_grid.fit(X_train_resampled, y_train_resampled)
    print("\nBest XGBoost Parameters:")
    print(xgb_grid.best_params_)

    # LightGBM GridSearch
    lgb_grid = GridSearchCV(
        lgb_model, lgb_param_grid, scoring='roc_auc', cv=cv, verbose=1, n_jobs=-1
    )
    lgb_grid.fit(X_train_resampled, y_train_resampled)
    print("\nBest LightGBM Parameters:")
    print(lgb_grid.best_params_)

    # Select the best model
    best_model = xgb_grid.best_estimator_ if xgb_grid.best_score_ >= lgb_grid.best_score_ else lgb_grid.best_estimator_
    model_name = "XGBoost" if xgb_grid.best_score_ >= lgb_grid.best_score_ else "LightGBM"
    print(f"\nSelected Model: {model_name}")

    # --- Step 8: Feature Selection Using SHAP ---
    print("\nPerforming feature selection with SHAP...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_resampled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    
    # Select top features (e.g., top 80% of importance)
    threshold = feature_importance.quantile(0.15)
    selected_features = feature_importance[feature_importance >= threshold].index.tolist()
    print(f"Selected {len(selected_features)} features: {selected_features}")

    # Update datasets with selected features
    X_train_selected = X_train_resampled[selected_features]
    X_test_selected = X_test_processed[selected_features]

    # Retrain best model on selected features
    best_model.fit(X_train_selected, y_train_resampled)

    # --- Step 9: Evaluate the Model ---
    print("\nEvaluating the model...")
    y_pred = best_model.predict(X_test_selected)
    y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest Set AUC-ROC Score: {auc:.4f}")

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not High Stress', 'High Stress']))

    print("\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Not High Stress', 'Predicted High Stress'],
                yticklabels=['Actual Not High Stress', 'Actual High Stress'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Step 10: SHAP Analysis ---
    print("\nPerforming SHAP analysis...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_selected)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_test_selected, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values, X_test_selected, show=False)
    plt.tight_layout()
    plt.show()

    top_feature_index = np.argmax(np.abs(shap_values).mean(axis=0))
    top_feature_name = X_test_selected.columns[top_feature_index]
    plt.figure()
    shap.dependence_plot(top_feature_index, shap_values, X_test_selected, show=False)
    plt.title(f"SHAP Dependence Plot for {top_feature_name}")
    plt.tight_layout()
    plt.show()

    print("\n--- Modeling and Interpretation Complete ---")