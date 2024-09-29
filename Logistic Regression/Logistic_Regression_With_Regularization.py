import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def Sigmoid(w , b , x):
    z = np.dot(w , x) + b
    return 1 / (1 + np.exp(-z)) # This is sigmoid function..



def logistic_loss(X , y , w , b , lamda): # X is 2D array and W is 1D array 
    m = X.shape[0]
    loss = 0
    for i in range(m):
        z = Sigmoid(w , b , X[i])
        loss += -y[i] * np.log(z) - (1 - y[i])*np.log(1 - z)
    loss = loss / m
    loss += lamda*np.sum(w**2) / (2*m) # This is regularization term. (L2 Regularization -> Ridge)
    return loss


def Compute_Gradient(X , y , w , b , lamda):
     n = X.shape[1]
     m = X.shape[0]
     d_w = np.zeros(n)
     d_b = 0
     for i in range(m):
         
         for j in range(n):
             d_w[j] += (Sigmoid(w , b , X[i])-y[i]) * X[i][j]
         d_b += (Sigmoid(w , b , X[i]) - y[i])

     d_w = d_w / m + (lamda * np.sum(d_w)) / m # This is regularized term.
     d_b = d_b / m         
     return (d_w , d_b)



def Gradient_Descent(X , y , w , b , alpha):
    iter = 10000
    lamda = 10
    for i in range(iter):
        d_w , d_b = Compute_Gradient(X , y , w , b,lamda)
        w = w - alpha * d_w
        b = b - alpha * d_b
    return w , b 



def Predict(w , b ,input):
    output = []
    for i in test_x_scaled:
         output.append(Sigmoid(w,b,i))
    
    prediction = [1 if x>=0.5 else 0 for x in output]
    return prediction




data = [
    [22, 50000], # This data needs to be scaled .
    [25, 54000],
    [28, 61000],
    [32, 65000],
    [35, 70000],
    [40, 72000],
    [42, 75000],
    [45, 80000],
    [50, 85000],
    [55, 90000]
]

scaler = StandardScaler()
train_x = np.array(data)
train_x_scaled = scaler.fit_transform(train_x)

train_y = np.array([0,0,0,1,1,1,1,1,1,1])
w , b = Gradient_Descent(train_x_scaled , train_y,np.zeros(2),0,0.01)

test_data = [
    [23, 51000, 0],
    [29, 62000, 0],
    [38, 67000, 1],
    [41, 76000, 1],
    [48, 83000, 1]
]
test_data = np.array(test_data)
test_x = test_data[:,:2]
test_y = test_data[:,2]

test_x_scaled = scaler.transform(test_x)

prediction = Predict(w , b , test_x_scaled)
print(f"Test Data prediction is {prediction}")
acurracy = accuracy_score(test_y,prediction)
print(f"Test Data Accuracy of algorithm is: {acurracy*100}%")

# Getting 100 % accuracy could be sign of overfitting. 
# Lets check the accuracy of train_data.

print(f"Train Data Prediction: {Predict(w , b , train_x_scaled)}")
train_predict = Predict(w , b , train_x_scaled)


