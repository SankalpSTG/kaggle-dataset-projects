import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# Read Data
df = pd.read_csv("house-price-prediction.csv")

print(df.head())
df = df.drop("Id", axis=1)

#Assume absent data
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

#Assume numeric absent values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

#Assume string absent values
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

print(df.isnull().sum())

# Boxplot to see the effect of categorical features like 'Location', 'Condition'
plt.figure(figsize=(10, 6))
# sns.boxplot(X="Area", y='Price', data=df)
sns.boxplot(y='Price', data=df)
plt.xticks(rotation=90)
plt.show()

averagePricePerLocation = df.groupby('Location')['Price'].transform('mean')
df['LocationRatio'] = df['Price'] / averagePricePerLocation

# ENCODING STRING VALUES
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

encoders = {}

for col in categorical_cols:
  unique_values = df[col].nunique()
  if unique_values <= 2:
    df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(df[col])
    print(f'Encoded {col} as 0/1.')
  else:
    df = pd.get_dummies(df, columns=[col], drop_first=True)
    print(f'Encoded {col} with One-Hot Encoding.')


X = df.drop(['Price'], axis=1)
y = df['Price']

# SPLITTING TEST DATA AND TRAIN DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LINEAR REGRESSION
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# RANDOM FOREST REGRESSOR
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("RANDOM FOREST RESULTS: ")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
print("R^2 Score:", r2_score(y_test, y_pred_rf))

# FINDING OUT IMPORTANT FEATURES AMONG ALL
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.show()

# XGB
xgb_model = XGBRegressor(random_state=42)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Results: ")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_xgb))
print("R^2 Score:", r2_score(y_test, y_pred_xgb))

#LIGHTGB
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
print("LightGBM Results: ")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_lgb))
print("R^2 Score:", r2_score(y_test, y_pred_lgb))

#SVR WITH TUNED HYPERPARAMETERS
param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.2],
    'svr__kernel': ['linear', 'rbf', 'poly']
}
svr_pipeline = make_pipeline(StandardScaler(), SVR())
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred_svr = best_model.predict(X_test)
print("Best parameters found: ", grid_search.best_params_)
print("SVR Results: ")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_svr))
print("R^2 Score:", r2_score(y_test, y_pred_svr))
