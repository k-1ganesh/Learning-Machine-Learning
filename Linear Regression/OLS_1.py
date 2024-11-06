import numpy as np

# OLS -> Ordinal Least Square.
# This Problem is solved using OLS Method. Directly using formula.
# Lets create class 
class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X_train , y_train):
        num = 0 
        den = 0
        for i in range(X_train.shape[0]):
            num += (y_train[i] - y_train.mean()) * (X_train[i] - X_train.mean())
            den += (X_train[i] - X_train.mean)**2
        self.w = num / den
        self.b = y_train.mean() - self.w * X_train.mean()

    def predict(self,X_test):
        return self.w * X_test + self.b
    
# Lets create object 
model = LinearRegression()
model.fit() # Used to Train the model
model.predict() # Used to predict the result.