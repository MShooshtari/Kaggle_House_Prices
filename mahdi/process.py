from sklearn.base import BaseEstimator, TransformerMixin

# Artificial Neural Networks Class definition
import keras
from keras.models import Sequential
from keras.layers import Dense

class ANN(BaseEstimator, TransformerMixin):

    def __init__(self, optimizer='adam', loss='mean_squared_error', batch_size=100, nb_epoch=2000):
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
    
    def fit(self, X, y=None):
        regressor = Sequential()
        # Adding the input layer and the first hidden layer 
        # Number of nodes in each layer: Average of number of input nodes and ouput nodes
        # In our case, we have 42 input nodes and 2 classes, therfore 1 output node, so the average is 22 or 21.
        (rows, cols) = X.shape
        numberOfInputNodes = cols
        numberOfOutputNodes = 1
        numberOfMiddleNodes = int((numberOfInputNodes + numberOfOutputNodes)/2)
        regressor.add(Dense(input_dim=numberOfInputNodes, output_dim=numberOfMiddleNodes, activation='relu'))
        regressor.add(Dense(output_dim=numberOfMiddleNodes, activation='relu'))
        regressor.add(Dense(output_dim=numberOfMiddleNodes, activation='relu'))
        regressor.add(Dense(output_dim=numberOfOutputNodes, activation='linear'))

        # Compiling the ANN
        regressor.compile(optimizer=self.optimizer, loss=self.loss)
        # Fitting the ANN to the training set
        regressor.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch)
        self.regressor = regressor
        
        return self
    
    def transform(self, X):
        pass
    
    def predict(self, X):
      return self.regressor.predict(X)


