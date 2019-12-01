import json 
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin

# Convert Categorical columns to Numeric
class CatToNum(BaseEstimator, TransformerMixin):

    def __init__(self, dict_address, continuesVars, descreteVars_Ordinal, descreteVars_Nominal):
        self.continuesVars = continuesVars
        self.descreteVars_Ordinal = descreteVars_Ordinal
        self.descreteVars_Nominal = descreteVars_Nominal
        
        fileName = dict_address
        f = open(fileName,'r')
        self.conversion_dict = json.loads(f.read())
        

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        des_nom_DF = X[self.descreteVars_Nominal]
        # Map Nominal Categorical data to Numerical
        cat_nom_DF = des_nom_DF.fillna('NULL').astype(str)
        self.ce_binary = ce.BinaryEncoder()
        self.ce_binary.fit(cat_nom_DF)
        return self

    def transform(self, X):
        totalDF = X.copy()
        conDF = X[self.continuesVars]
        des_ord_DF = X[self.descreteVars_Ordinal]
        des_nom_DF = X[self.descreteVars_Nominal]
        
        # Map Ordinal Categorical data to Numerical
        cat_ord_DF_numerical = des_ord_DF.copy()
        for feature in self.conversion_dict:
            temp_dict = self.conversion_dict[feature]
            if ('NA' in temp_dict): # Replace 'NA' with np.nan
                temp_dict[np.nan] = temp_dict.pop('NA')
            cat_ord_DF_numerical[feature] = des_ord_DF[feature].map(temp_dict)
            
        totalDF[self.descreteVars_Ordinal] = cat_ord_DF_numerical
            
        # Map Nominal Categorical data to Numerical
        cat_nom_DF = des_nom_DF.fillna('NULL').astype(str)
        cat_nom_DF_numerical = self.ce_binary.transform(cat_nom_DF)
        totalDF = pd.concat([totalDF, cat_nom_DF_numerical], axis=1)
        totalDF.drop(self.descreteVars_Nominal, axis=1, inplace=True)
        cols_to_drop = [x for x in list(cat_nom_DF_numerical) if ('_0' in x)]
        totalDF.drop(cols_to_drop, axis=1, inplace=True)

        return totalDF


# Clean Numeric Data
class CleanNum(BaseEstimator, TransformerMixin):

    def __init__(self, measure='mean', drop_thresh=0.8):
        self.measure = measure
        self.drop_thresh = drop_thresh

    def fit(self, X, y=None):
        train_df = X.copy().astype(float)
        # Remove columns with dominance bigger than a threshold
        self.drop_list = []
        for feature in train_df:
            col_df = train_df[feature]
            count_nan = col_df.isnull().sum()
            nan_ratio = count_nan/len(col_df)   
            repeats = train_df.pivot_table(index=[feature], aggfunc='size').sort_values()
            max_repeat_ratio = repeats.max()/len(col_df)
            if (nan_ratio>self.drop_thresh or max_repeat_ratio>self.drop_thresh):
                self.drop_list.append(feature)

        # Replace null values with average (or mode) of the train columns
        self.cols_average = train_df.mean(axis = 0)
        self.cols_mode = train_df.mode(axis = 0)
        
        return self

    def transform(self, X):
        totalDF = X.copy().astype(float)
        
        # Replace null values with average (or mode) of the train columns
        for col in totalDF:
            if (self.measure=='mean'):
                totalDF[col] = totalDF[col].fillna(self.cols_average[col])
            if (self.measure=='mode'):
                totalDF[col] = totalDF[col].fillna(self.cols_mode[col])
        
        # Remove columns with dominance bigger than a threshold
        totalDF.drop(self.drop_list, axis=1, inplace=True)

        return totalDF


# Outlier Detection
class OutlierDetection(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

        # Function to Detection Outlier on one-dimentional datasets.
    def find_anomalies(self,data):
        anomalies_idx = []
        # Set upper and lower limit to 3 standard deviation
        data_std = data.std()
        data_mean = data.mean()
        anomaly_cut_off = data_std * 3

        lower_limit  = data_mean - anomaly_cut_off 
        upper_limit = data_mean + anomaly_cut_off
        # Generate outliers
        for idx in range(len(data)):
            outlier = data[idx]
            if outlier > upper_limit or outlier < lower_limit:
                anomalies_idx.append(idx)
        return anomalies_idx

    def fit(self, X, y=None):
        train_df = X.copy()
        outliers_full_list = []
        for feature in train_df:
            outliers_full_list = outliers_full_list + self.find_anomalies(train_df[feature].values)

        self.outliers_unique = list(set(int(outliers_full_list)))
        return self

    def transform(self, X):
        totalDF = X.copy()
        totalDF.drop(self.outliers_unique, inplace=True)
        return totalDF


# Data Transformation (Box-Cox transformation)
from scipy.special import boxcox1p
from scipy.stats import norm, skew

class DataTransformation(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, X, y=None, thresh=0.75, lam=0.15):
        self.lam = lam
        train_df = X.copy()
        # Check the skew of all numerical features
        skewed_feats = train_df.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness = skewness[abs(skewness) > thresh]
        self.skewed_features_idx_list = skewness.index
        return self
    
    def transform(self, X):
        totalDF = X.copy()
        for feat_idx in self.skewed_features_idx_list:
            totalDF[feat_idx] = boxcox1p(totalDF[feat_idx], self.lam)
        return totalDF