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
    iter = 10000
    for i in range(iter):
        d_w , d_b = Gradient(x , y , w , b)
        w = w - alpha * d_w
        b = b - alpha * d_b
    return w , b

def zNormalization(x):
    mu = np.mean(x , axis = 0)
    std = np.std(x , axis = 0)
    x_norm = (x - mu) / std
    return x_norm , mu , std


x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) # Data Needs to be normalized first
x_norm,mu,std = zNormalization(x)
y = np.array([460, 232, 178])  
w , b = Gradient_Descent(x_norm,y,np.zeros(4),0,0.01)
input = np.array( [2104,5,1,45])
print(Predict((input - mu) / std , w , b)) # At the time of prediction we again need to normalize the data
print(x_norm)




