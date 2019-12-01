from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
 

def load_model(model_name):
	if (model_name == 'KernelRidge'):
		return (KernelRidge())
	if (model_name == 'ElasticNet'):
		return (ElasticNet())
	if (model_name == 'Lasso'):
		return (Lasso())
	if (model_name == 'GradientBoostingRegressor'):
		return (GradientBoostingRegressor())
	if (model_name == 'BayesianRidge'):
		return (BayesianRidge())
	if (model_name == 'LassoLarsIC'):
		return (LassoLarsIC())
	if (model_name == 'RandomForestRegressor'):
		return (RandomForestRegressor())
	if (model_name == 'XGBRegressor'):
		return (XGBRegressor())
	if (model_name == 'ANN'):
		from mahdi import process		
		return (process.ANN())

