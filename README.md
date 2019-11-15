# Kaggle_House_Prices

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Practice Skills Creative feature engineering Advanced regression techniques like random forest and gradient boosting

Link to the Kaggle Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview


**Pre Process**
- [X] Categorical vs Numerical Data
- [X] Clean the numeric converted data (Null, Range, etc...)
- [X] Outlier detection
- [X] Skewed data transformation
- [X] Normalization
- [X] Variable correlation
- [X] Dimensionality reduction
   - [X] PCA
   - [ ] Incremental, decremental feature selection?


**Main Process**
- [X] Linear regression
- [X] XGBoost regression
	- [ ] Grid Search
- [X] Random Forest regression
- [X] Artificial Neural Network regression
- [X] Lasso Lars IC
- [X] Elastic Net
- [X] Bayesian Ridge
- [X] Gradient Boosting
- [X] Combining (averaging) results
	- [X] ANN & RF
- [X] Ensemble
- [X] Stacking

**Results**

| Methods | File | Score |
| ------ | ------ | ------ |
| Linear Regression | Lin_reg_WO_PCA | 0.61590 |
| XGBoost | XGBoost_reg_WO_PCA | 0.46689 |
| Random Forest | RF_reg_WO_PCA | 0.172322 |
| ANN | ANN_reg_WO_PCA | 0.16323 |
| Average of ANN & RF | ANN_&\_RF_WO_PCA | 0.15605 |
| LassoLarsIC | LassoLarsIC | 0.17450 |
| ElasticNet_transformed | ElasticNet_transformed | 0.82561 |
| ElasticNet | ElasticNet | 0.20401 |
| BayesianRidge_transformed | BayesianRidge_transformed | 0.82334 |
| BayesianRidge | BayesianRidge | 0.18784 |
| GradientBoostingRegressor_transformed | GradientBoostingRegressor_transformed | 0.13962 | 
| XGBRegressor_transformed | XGBRegressor_transformed | 0.14549 |
| ---- | ---- | ---- |
| Stacking (Metamodel = RF) | RandomForestRegressor_final | 0.13485 |
| Stacking (Metamodel = XGB) | XGBRegressor_final | 0.13672 |
| Stacking (Average of Metamodels) | Average_of_MetaModels | 0.13827 |
