import numpy as np

def Gradient(x,y,w,b):
    m = len(x)
    d_b = 0
    d_w = 0
    for i in range(m):
        d_w = d_w + (w*x[i] + b - y[i])*x[i]
        d_b = d_b + (w*x[i] + b - y[i])
    d_w = d_w / m
    d_b = d_b / m
    return d_w , d_b

def Predict(x , w , b):
    return x * w + b

def Gradient_descent(x , y , w , b):
    iter = 10000
    alpha = 0.0001
    for i in range(iter):
        d_w , d_b = Gradient(x , y ,w , b)
        w = w - alpha * d_w
        b = b - alpha * d_b
    return w , b

x = np.array([10 , 20 , 50 , 100 , 70 , 200])
y = np.array([100 , 150 , 400, 1200 , 350 , 1500])

w , b = Gradient_descent(x , y , 0 , 0)
print(Predict(150 , w , b))
print(w , b)
        
