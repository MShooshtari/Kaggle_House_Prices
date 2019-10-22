# Kaggle_House_Prices

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Practice Skills Creative feature engineering Advanced regression techniques like random forest and gradient boosting

Link to the Kaggle Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview


**Pre Process**
- [X] Categorical vs Numerical Data
- [X] Clean the numeric converted data (Null, Range, etc...)
- [X] Outlier detection
- [ ] Skewed data transformation
- [X] Normalization
- [X] Variable correlation
- [X] Dimensionality reduction
   - [X] PCA
   - [ ] Incremental, decremental feature selection?


**Main Process**
- [X] Linear regression
- [X] XGBoost regression
	- [ ] Grid Search
- [ ] Random Forest regression
- [ ] Artificial Neural Network regression

**Results**

| Methods | File | Normalized without PCA | File | Normalized with PCA |
| ------ | ------ | ------ | ------ | ------ |
| Linear Regression | Lin_reg_WO_PCA.csv | 0.61590 | Lin_reg_W_PCA.csv | 0.61248 |
| XGBoost | XGBoost_reg_WO_PCA | 0.46689 | XGBoost_reg_W_PCA | 0.48262 |
| Random Forest | - | - | - | - |
| ANN | ANN_reg_WO_PCA | 0.16323 | ANN_reg_W_PCA | 0.27617 |
