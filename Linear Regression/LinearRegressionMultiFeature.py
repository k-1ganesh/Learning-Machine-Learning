import numpy as np

def Predict(x , w , b): # In this x is 1D array
    return np.dot(x , w) + b


def Gradient(x , y , w , b): # In this case x is 2D array
    m = x.shape[0]
    d_w = np.zeros(x.shape[1])
    d_b = 0
    for i in range(m):
        for j in range(x.shape[1]):
            d_w[j] += (np.dot(w , x[i]) + b - y[i])*x[i][j]
        d_b += (np.dot(w , x[i]) + b - y[i])
    d_w = d_w / m
    d_b = d_b / m
    return d_w , d_b

def Gradient_Descent(x , y , w , b , alpha): # In this case x is 2D array
    iter = 1000
    for i in range(iter):
        d_w , d_b = Gradient(x , y , w , b)
        w = w - alpha * d_w
        b = b - alpha * d_b
    return w , b

x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) 
y = np.array([460, 232, 178])  
w , b = Gradient_Descent(x,y,np.zeros(4),0,5.0e-7)
input = np.array( [2104,5,1,45])
print(Predict(input , w , b))





