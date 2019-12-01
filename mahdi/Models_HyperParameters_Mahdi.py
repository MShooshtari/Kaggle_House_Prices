hyperparams = dict()
hyperparams['KernelRidge'] = {'alpha': 0.1, 'coef0': 100, 'degree': 1, 'gamma': None, 'kernel': 'polynomial'}
hyperparams['ElasticNet'] = {'alpha': 0.001, 'copy_X': True, 'l1_ratio': 0.6, 'fit_intercept': True, 'normalize': False, 'precompute': False, 'max_iter': 300, 'tol': 0.001, 'selection': 'random', 'random_state': None}
hyperparams['Lasso'] = {'alpha': 0.0005, 'copy_X': True, 'fit_intercept': True, 'normalize': False, 'precompute': False, 'max_iter': 300, 'tol': 0.01, 'selection': 'random', 'random_state': None}
hyperparams['GradientBoostingRegressor'] = {'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 3, 'min_samples_split': 0.0025, 'min_samples_leaf': 5}
hyperparams['BayesianRidge'] = {'n_iter': 200, 'tol': 0.00001, 'alpha_1': 0.00000001, 'alpha_2': 0.000005, 'lambda_1': 0.000005, 'lambda_2': 0.00000001, 'copy_X': True}
hyperparams['LassoLarsIC'] = {'criterion': 'aic', 'normalize': True, 'max_iter': 100, 'copy_X': True, 'precompute': 'auto', 'eps': 0.000001}
hyperparams['RandomForestRegressor'] = {'n_estimators': 50, 'max_features': 'auto', 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 2}
hyperparams['XGBRegressor'] = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 300, 'booster': 'gbtree', 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 0.7, 'max_delta_step': 0, 'min_child_weight': 1, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.2, 'scale_pos_weight': 1}
hyperparams['ANN'] = {'optimizer': 'adam', 'loss': 'mean_squared_error'}

