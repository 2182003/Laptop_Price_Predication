# Laptop Price Prediction Project



# Problem Statement

Many users face difficulty determining the fair price of a laptop based on its specifications. This project uses a dataset of laptops with features such as company, type, screen size, CPU, RAM, storage, GPU, operating system, and weight to predict laptop prices. The goal is to help users estimate the price of a laptop configuration before buying.

# Dataset

* The dataset contains information about laptops, including:

* Company: Laptop brand

* TypeName: Type of laptop (Ultrabook, Gaming, Notebook, etc.)

* Inches: Screen size

* ScreenResolution: Screen resolution and features

* Cpu: Processor details

* Ram: RAM in GB

* Memory: Storage details (HDD, SSD, Flash)

* Gpu: Graphics card

* OpSys: Operating system

* Weight: Laptop weight

* Price: Target variable

* Note: Some preprocessing is done to extract numeric values and create new features like PPI (Pixels Per Inch), Touchscreen, IPS display, and Memory split.

## Stepwise Approach

# 1. Data Loading & Cleaning

Loaded CSV dataset using pandas.

Dropped unnecessary columns.

Checked for missing values and duplicates.

Converted columns like Ram and Weight to numeric types.

# 2. Exploratory Data Analysis (EDA)

Univariate Analysis:

Distribution of Price, screen size, and laptop count by company/type.

Boxplots to identify outliers in price.

Bivariate Analysis:

Price vs Company, TypeName, RAM, Memory, Screen size, CPU, GPU, OS.

Multivariate Analysis:

Correlation between numeric features and Price.

Pairplots for visualizing numeric feature relationships.

Insights:

Apple laptops have higher prices.

Higher RAM, better CPU, SSD storage, dedicated GPU increase prices.

Screen size, resolution, and weight also impact price.

# 3. Feature Engineering

Extracted features from ScreenResolution (Touchscreen, IPS, X_res, Y_res, PPI).

Processed Cpu column into brands (Intel, AMD, etc.).

Split Memory into HDD, SSD, Flash_Storage features.

Extracted GPU brand.

Consolidated operating systems into categories (Windows, Mac, Other).

Dropped redundant columns after feature extraction.

# 4. Data Preprocessing

Converted categorical variables using One-Hot Encoding.

Log transformation applied to Price for normalization.

Split dataset into train and test sets (15% test).

# 5. Model Building

Tested multiple regression models:

Linear Models: Linear Regression, Ridge, Lasso

K-Nearest Neighbors: KNN Regressor

Tree-based Models: Decision Tree, Extra Trees

Ensemble Models: Random Forest, Gradient Boosting, AdaBoost, Voting Regressor, Stacking Regressor

XGBoost Regressor

Pipelines were used with preprocessing + model training.

Hyperparameter tuning was performed using RandomizedSearchCV.

* get best R2 score by XGBoost

# 6. Model Evaluation

Evaluated using R² Score and Mean Absolute Error (MAE).

Among all models tested, XGBoost Regressor achieved the best R² score, showing it had the strongest predictive capability.



# 7. Model Export

Final trained model (pipe) and processed dataframe (df) exported using pickle for future use.

import pickle

pickle.dump(df, open('df.pkl', 'wb'))

pickle.dump(pipe, open('pipe.pkl', 'wb'))

# 8. Usage

Load the model using pickle and make predictions on new laptop configurations.

import pickle
import pandas as pd
import numpy as np

## Load trained pipeline
pipe = pickle.load(open('pipe.pkl','rb'))


## Predict price
predicted_price = np.exp(pipe.predict(sample_input))  # inverse log transform

Technologies Used:Python


Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Machine Learning Models: Linear Regression, Ridge, Lasso, KNN, Decision Trees, Random Forest, Gradient Boosting, AdaBoost, XGBoost, Voting & Stacking Regressors


# Conclusion

This project predicts laptop prices based on features and configurations.

Users can estimate fair prices before purchasing.

XGBoost Regressor achieved the best R² score, demonstrating high predictive performance.
