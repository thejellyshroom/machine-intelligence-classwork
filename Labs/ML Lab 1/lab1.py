'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Lab 1
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

'''Part 1, importing data'''
df = pd.read_csv("Labs/ML Lab 1/car details v4.csv")

#print the first 5 rows of the dataframe and the last 5 rows of the dataframe
print(df.head())
print(df.tail())

#print the shape of the dataframe (number of rows and columns)
print(df.shape)

#print the column names and data types of the columns
print(df.columns)
print(df.dtypes)

#make sure there are no missing values in each column
print("Missing values:")
if df.isnull().sum().any():
    print("There are missing values in the dataframe. Skipping missing values of row and column: " + str(df.isnull().sum()))
    #skip missing values
    df = df.dropna()
else:
    print("There are no missing values in the dataframe.")

#make sure there are no duplicate rows
print("Duplicate rows:")
duplicates = df.duplicated()
if duplicates.any():
    print(f"Found {duplicates.sum()} duplicate rows in the dataframe. Removing duplicates.")
    df = df.drop_duplicates()
else:
    print("There are no duplicate rows in the dataframe.")


'''Part 2, creating new attributes'''
print("\n\nCreating new attributes:")

# Car age (current year - manufacturing year)
current_year = 2025
df['car_age'] = current_year - df['Year']
print("Added 'car_age' attribute")

# Average kilometers driven per year
df['km_per_year'] = df['Kilometer'] / df['car_age']
print("Added 'km_per_year' attribute")

# Extract numeric values (only ints)
df['engine_cc'] = df['Engine'].str.extract(r'(\d+)').astype(float)
print("Added 'engine_cc' attribute")

df['power_bhp'] = df['Max Power'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
print("Added 'power_bhp' attribute")

df['torque_nm'] = df['Max Torque'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
print("Added 'torque_nm' attribute")

# Calculate power-to-engine ratio
df['power_to_engine_ratio'] = df['power_bhp'] / df['engine_cc']
print("Added 'power_to_engine_ratio' attribute")

# volume (length * width * height)
df['volume_m3'] = (df['Length'] * df['Width'] * df['Height']) / 1000000000  # Convert from cubic mm to cubic m
print("Added 'volume_m3' attribute")

# Price per CC
df['price_per_cc'] = df['Price'] / df['engine_cc']
print("Added 'price_per_cc' attribute")

# Print sample
print("\nSample of new features:")
print(df[['car_age', 'km_per_year', 'engine_cc', 'power_bhp', 'torque_nm',
          'power_to_engine_ratio', 'volume_m3', 'price_per_cc']].head())


'''Part 3, converting categorical data to numeric'''
print("\nConverting categorical data to numeric:")

# Separate features into different encoding groups
# One-hot encoding for nominal categories (no inherent order)
onehot_columns = ['Make', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Seller Type', 'Drivetrain']

# Ordinal encoding for ordered categories
ordinal_columns = ['Owner']
owner_categories = [['First', 'Second', 'Third', 'Fourth', 'UnRegistered Car']]

# Numerical columns (need scaling)
numerical_columns = ['Year', 'Price', 'Kilometer', 'engine_cc', 'power_bhp', 'torque_nm',
                    'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity',
                    'car_age', 'km_per_year', 'power_to_engine_ratio', 'volume_m3', 'price_per_cc']


'''Part 4, scaling data with standard scaler and transforming data with column transformer'''
# Create the column transformer with scaling for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('scaler', StandardScaler())]), numerical_columns),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), onehot_columns),
        ('ordinal', OrdinalEncoder(categories=owner_categories), ordinal_columns)
    ])

# Fit and transform the data
X_encoded = preprocessor.fit_transform(df)

# Get feature names after encoding
onehot_feature_names = []
for i, column in enumerate(onehot_columns):
    categories = preprocessor.named_transformers_['onehot'].categories_[i]
    onehot_feature_names.extend([f"{column}_{cat}" for cat in categories])

# Combine all feature names
feature_names = numerical_columns + onehot_feature_names + ['owner_encoded']


# Convert to DataFrame with proper column names
df_encoded = pd.DataFrame(X_encoded, columns=feature_names)

print("\nShape of data after encoding:", df_encoded.shape)
print("\nSample of encoded features (first few columns):")
print(df_encoded.iloc[:, :10].head())


'''Part 5, analyzing correlations'''
print("\nAnalyzing correlations between numerical features:")
numerical_df = df[numerical_columns]
correlation_matrix = numerical_df.corr()

# Create a heatmap of correlations
plt.figure(figsize=(15, 12))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(numerical_columns)), numerical_columns, rotation=45, ha='right')
plt.yticks(range(len(numerical_columns)), numerical_columns)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('Labs/ML Lab 1/correlation_matrix.png')
plt.close()

# Print the most significant correlations with Price
price_correlations = correlation_matrix['Price'].sort_values(ascending=False)
print("\nTop correlations with Price:")
print(price_correlations)


'''Part 6, training and testing the model'''
print("\nTraining and testing the model:")

# Separate features (X) and target variable (y)
X = df_encoded.drop('Price', axis=1)  # Remove Price from features
y = df_encoded['Price']  # Target variable

# 80-20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation for both models
print("\nPerforming 5-fold cross-validation:")

# Linear Regression
linear_cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='r2')
print("\nLinear Regression CV Scores:")
print(f"R² scores for each fold: {linear_cv_scores}")
print(f"Mean R² score: {linear_cv_scores.mean():.4f} (+/- {linear_cv_scores.std() * 2:.4f})")

# Random Forest
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print("\nRandom Forest CV Scores:")
print(f"R² scores for each fold: {rf_cv_scores}")
print(f"Mean R² score: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

# Train final models on the entire training set
print("\nTraining final models on entire training set:")

# Linear Regression
linear_model.fit(X_train, y_train)
linear_train_pred = linear_model.fit(X_train, y_train).predict(X_train)
linear_test_pred = linear_model.predict(X_test)

# Random Forest
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.fit(X_train, y_train).predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Evaluate models
print("\nModel Performance Metrics:")
print("\nLinear Regression:")
print(f"Training R² score: {r2_score(y_train, linear_train_pred):.4f}")
print(f"Testing R² score: {r2_score(y_test, linear_test_pred):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, linear_train_pred)):.2f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, linear_test_pred)):.2f}")

print("\nRandom Forest:")
print(f"Training R² score: {r2_score(y_train, rf_train_pred):.4f}")
print(f"Testing R² score: {r2_score(y_test, rf_test_pred):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, rf_train_pred)):.2f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, rf_test_pred)):.2f}")

# Plot actual vs predicted values for both models
plt.figure(figsize=(15, 6))

# Linear Regression plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, linear_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: Actual vs Predicted Prices')

# Random Forest plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted Prices')

plt.tight_layout()
plt.savefig('Labs/ML Lab 1/prediction_comparison.png')
plt.close()

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importance.head(10))
