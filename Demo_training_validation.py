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
# Some extra information required for running the preprocess steps are mentioned in Extra_Information_Mahdi.py
	# List of descrete nomainal features
	# List of descrete ordinal features
	# Dictionary address for converting nominal categorical to ordinal
from mahdi import Extra_Information as EIM
descreteVars_Nominal = EIM.descreteVars_Nominal
descreteVars_Ordinal = EIM.descreteVars_Ordinal
continuesVars = list(set(list(train_df)) - set(descreteVars_Nominal) - set(descreteVars_Ordinal) - set(resultVar))
dict_address = EIM.dict_address

# Preprocess steps:
	# CatToNum: Converting categorical variables to numeric. Data conversion is different according the nominal-vs-ordinal type of features. 
		# Input: Original data
		# Output: Numeric data
	# CleanNum: Clean numeric table from None, nan and replace those with average or mode.
		# Input: Numeric data (It will give you an in cases such as categorical data)
		# Output: Numeric data (without nulls)
	# Data Transformation: Transforming skewed features using box-cox transformation
		# Input: Numeric data (without outliers)
		# Output: Numeric data (transformed)
from mahdi.preprocess import CatToNum
from mahdi.preprocess import CleanNum
from mahdi.preprocess import DataTransformation
from sklearn.preprocessing import RobustScaler # It can be replaced by other types of scalers such as min-max
preprocess_steps = []
preprocess_steps += [('cat_to_num', CatToNum(dict_address, continuesVars, descreteVars_Ordinal, descreteVars_Nominal))]
preprocess_steps += [('clean_num', CleanNum())]
preprocess_steps += [('data_transformation', DataTransformation())]
preprocess_steps += [('robust_scalar', RobustScaler())]


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
from mahdi import Models_HyperParameters as MHM
from mahdi import Load_Models as LMM
from sklearn.pipeline import Pipeline

# Note: ANN model will use Keras libraries and is compatible only with GPU devices. If you have one, you can uncomment the next line
models_names_list = ['KernelRidge', 'ElasticNet', 'Lasso', 'GradientBoostingRegressor', 'BayesianRidge', 'LassoLarsIC', 'RandomForestRegressor', 'XGBRegressor', 'ANN']
# models_names_list += ['ANN']

# Selecting a model
# You can replace this with any other model from the list above. Or have a loop over them or randomly select one, your choice :)
model_name = models_names_list[3] 

# Loading the model
model = LMM.load_model(model_name)
params = MHM.hyperparams[model_name]

# Make the sklearn pipline based on the preprocess steps and the regression model
pipeline_steps = preprocess_steps + [('regression_model', model.set_params(**params))]
pipeline = Pipeline(pipeline_steps)


# Train the model based on the pipeline
pipeline.fit(X_train, y_train)


# Predict the validation set
Y_pred = pipeline.predict(X_val)
print(Y_pred)
print(Y_pred.shape)

# Predict the test set
Y_pred = pipeline.predict(test_df)
print(Y_pred)
print(Y_pred.shape)