import numpy as np

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
model.fit()
model.predict()

