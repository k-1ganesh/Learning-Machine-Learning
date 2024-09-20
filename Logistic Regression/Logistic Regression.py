import numpy as np

def Sigmoid(w , b , x):
    z = np.dot(w , x) + b
    return 1 / (1 + np.exp(-z)) # This is sigmoid function..

def logistic_loss(X , y , w , b): # X is 2D array and W is 1D array 
    m = X.shape[0]
    loss = 0
    for i in range(m):
        z = Sigmoid(w , b , X[i])
        loss += -y * np.log(z) - (1 - y)*np.log(1 - z)
    return loss

def Compute_Gradient(X , y , w , b):
     n = X.shape[1]
     m = X.shape[0]
     d_w = np.zeros(n)
     d_b = 0
     for i in range(m):
         
         for j in range(n):
             d_w[j] += (Sigmoid(w , b , X[i])-y[i]) * X[i][j]
         d_b += (Sigmoid(w , b , X[i]) - y[i])

     d_w = d_w / m
     d_b = d_b / m         
     return (d_w , d_b)

def Gradient_Descent(X , y , w , b , alpha):
    iter = 10000
    for i in range(iter):
        d_w , d_b = Compute_Gradient(X , y , w , b)
        w = w - alpha * d_w
        b = b - alpha * d_b
    return w , b 

