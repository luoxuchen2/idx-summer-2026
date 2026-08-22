# California Residential Property Price Prediction

## Project Overview

This project predicts residential property sale prices in California using historical real estate transaction data and machine learning. The analysis focuses on identifying relationships between property characteristics, geographic information, and final sale prices.

The project includes data preprocessing, feature engineering, baseline modeling, advanced machine learning models, performance evaluation, and model refinement.

## Dataset Source

The dataset consists of California real estate listing and transaction records provided through IDX Exchange.

The analysis focuses on properties that meet the following conditions:

- `PropertyType`: `Residential`
- `PropertySubType`: `SingleFamilyResidence`

The target variable is:

- `ClosePrice`: The final sale price of a residential property.

The refined modeling dataset includes transactions from June 2025 through June 2026 and contains 392 engineered features.

The complete dataset is not included in the public repository because the source data contains proprietary real estate information.

## Data Preprocessing

The preprocessing workflow includes:

1. Combining monthly California real estate transaction files.
2. Filtering the dataset to residential single-family properties.
3. Reviewing missing values and removing columns with excessive missing data.
4. Preparing numerical and categorical variables for modeling.
5. Encoding categorical variables.
6. Reviewing extreme values and handling outliers.
7. Creating an engineered dataset for model development.
8. Splitting the data chronologically using the transaction month.

Examples of features considered include:

- Living area
- Number of bedrooms
- Number of bathrooms
- Lot size
- Year built
- Days on market
- Latitude and longitude
- City
- County
- ZIP code
- School district

The variables `ListPrice` and `OriginalListPrice` are excluded to reduce data leakage.

The final modeling dataset is saved as:

```text
output_csv/engineered.csv
```

## Training, Validation, and Testing

The project uses a chronological split to evaluate whether historical transaction data can predict future sale prices.

The final model uses a 12-month training window:

| Dataset | Time Period | Purpose |
| --- | --- | --- |
| Initial training data | June 2025–April 2026 | Train candidate model configurations |
| Validation data | May 2026 | Select the best hyperparameters |
| Full training data | June 2025–May 2026 | Train the final selected model |
| Test data | June 2026 | Evaluate final model performance |

Using the most recent month as the test set helps simulate a real-world prediction scenario.

## Models Tested

### Linear Regression

Linear regression was used as an initial baseline model to evaluate the relationship between property characteristics and sale prices.

### Decision Tree Regressor

A decision tree model was used to capture nonlinear relationships and interactions between property features.

### Random Forest Regressor

A random forest model was used to combine predictions from multiple decision trees and improve prediction stability.

### XGBoost Regressor

XGBoost was used as an advanced gradient boosting model to capture more complex relationships in the housing data.

The XGBoost model was refined by testing different combinations of:

- Maximum tree depth
- Learning rate
- Number of estimators
- Minimum child weight
- Row subsampling
- Feature subsampling
- L1 regularization
- L2 regularization

A total of 12 hyperparameter configurations were evaluated.

## Evaluation Metrics

The models were evaluated using:

- **R²:** Measures how much variation in property sale prices is explained by the model.
- **MAE:** Measures the average absolute prediction error in dollars.
- **RMSE:** Measures prediction error in dollars while giving greater weight to larger errors.
- **MAPE:** Measures the average absolute percentage prediction error.
- **MdAPE:** Measures the median absolute percentage prediction error.

The refinement notebook reports R², MAE, and RMSE. Additional percentage-based metrics are calculated in the evaluation notebook.

## Best Model Results

The final refined XGBoost model achieved the following results on June 2026 test data:

| Model | Test R² | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Previous XGBoost baseline | 0.4843 | $435,269.85 | $1,103,416.04 |
| Refined XGBoost model | 0.5544 | $447,720.20 | $1,025,668.90 |

The refined model improved test R² by 0.0701 and reduced RMSE by $77,747.14 compared with the previous XGBoost baseline.

However, MAE increased by $12,450.35, indicating that the model reduced larger prediction errors but did not improve the average absolute prediction error.

### Final XGBoost Hyperparameters

```python
XGBRegressor(
    objective="reg:squarederror",
    max_depth=3,
    learning_rate=0.05,
    n_estimators=300,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
```

The final model comparison is saved as:

```text
output_csv/enhanced_model_results.csv
```
