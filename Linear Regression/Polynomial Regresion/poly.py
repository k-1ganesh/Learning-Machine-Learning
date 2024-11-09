# This is used when data does not have linear behavior. 
# This non linear behaviour of data can be handelled by fitting polynomial curve.

import numpy as np
from sklearn.preprocessing import PolynomialFeatures 

class LinearRegression:
    def __init__(self):
        self.B = None
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self , X_train  ,y_train):
        # Lets modify X_train and add first column of ones.
        X = np.hstack( ( np.ones( (X_train.shape[0] , 1) ),X_train ) ) 
        Y = y_train 
        B1 = np.linalg.inv(np.dot(X.T , X))
        B2 = np.dot(X.T , Y)
        self.B = np.dot(B1 , B2)
        self.coef_ = self.B[1:]
        self.intercept_ = self.B[0]
    
    def predict(self,X_test):
        return np.dot(X_test , self.coef_) + self.intercept_

model = LinearRegression()

# Lets add polynomial term to the x_train.


# poly = PolynomialFeatures(degree=2)
# X_train_trans = poly.fit_transform(X_train)
# X_test_trans = poly.transform(X_test)

model.fit()
model.predict()

