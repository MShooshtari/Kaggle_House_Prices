# Installing requirements
# Upgrad your pip by typing the following command in your command propmt window:
# python -m pip install --upgrade pip

# Please install requirement llibraries first. You can find the full list in each folder.
# The requirement file is called requirements.txt and in order to install that you should type the following command in your command prompt:
# pip install requirements.txt

import pandas as pd 

# Load the data
train_file = 'house-prices-advanced-regression-techniques/train.csv'
train_df = pd.read_csv(train_file)
test_file = 'house-prices-advanced-regression-techniques/test.csv'
test_df = pd.read_csv(test_file)
resultVar = ['SalePrice']

# Divide the training data into training and validation sets
from sklearn.model_selection import train_test_split
X = train_df.drop(resultVar, axis=1)
y = train_df[resultVar]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)



##### Mahdi's approach
# Extra information:
# Some extra information required for running the preprocess steps is mentioned in Extra_Information_Mahdi.py
	# List of descrete nomainal features
	# List of descrete ordinal features
	# Dictionary address for converting categorical to nominal
# Preprocess steps:
	# CatToNum: Converting categorical variables to numeric. Data conversion is different according the nominal-vs-ordinal type of features. 
		# Input: Original data
		# Output: Numeric data
	# CleanNum: Clean numeric table from None, nan and replace those with average or mode.
		# Input: Numeric data (It will give you an in cases such as categorical data)
		# Output: Numeric data (without nulls)
	# Outlier Detection: Removing outliers that are more or less than 'n' times of variation (default 'n' is 3).
		# Input: Numeric data (without nulls)
		# Output: Numeric data (without outliers)
	# Data Transformation: Transforming skewed features using box-cox transformation
		# Input: Numeric data (without outliers)
		# Output: Numeric data (transformed)
# Available models:
	# 'KernelRidge': Scikit-learn Kernel Ridge regression algorithm
	# 'ElasticNet': Scikit-learn Elastic Net regression algorithm
	# 'Lasso': Scikit-learn Lasso regression algorithm
	# 'GradientBoostingRegressor': Scikit-learn Gradient Boosting regression algorithm
	# 'BayesianRidge': Scikit-learn Bayesian Ridge regression algorithm
	# 'LassoLarsIC': Scikit-learn LassoLarsIC regression algorithm
	# 'RandomForestRegressor': Scikit-learn Random Forest regression algorithm
	# 'XGBRegressor': XGBoost regression algorithm 
	# 'ANN': Artificial Neural Network model -> two middle layers, each with number_of_input_nodes/2 nodes.

# Extra information
from mahdi import Extra_Information_Mahdi as EIM
descreteVars_Nominal = EIM.descreteVars_Nominal
descreteVars_Ordinal = EIM.descreteVars_Ordinal
continuesVars = list(set(list(train_df)) - set(descreteVars_Nominal) - set(descreteVars_Ordinal) - set(resultVar))
dict_address = EIM.dict_address

# Preprocess steps
from mahdi.preprocess import CatToNum
from mahdi.preprocess import CleanNum
from mahdi.preprocess import OutlierDetection
from mahdi.preprocess import DataTransformation
from sklearn.preprocessing import RobustScaler # It can be replaced by other types of scalers such as min-max
preprocess_steps = [CatToNum(dict_address, continuesVars, descreteVars_Ordinal, descreteVars_Nominal), CleanNum(), OutlierDetection(), DataTransformation(), RobustScaler()]


# Available models
import Models_HyperParameters_Mahdi as MHM
models_list = ['KernelRidge', 'ElasticNet', 'Lasso', 'GradientBoostingRegressor', 'BayesianRidge', 'LassoLarsIC', 'RandomForestRegressor', 'XGBRegressor', 'ANN']
model = models_list[3]
hyperparams = MHM.hyperparams[model]

