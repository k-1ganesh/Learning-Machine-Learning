# Importing necessary Libraries
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
############################################################################################

# Creating Necessary Functions :
def Eucliedean_Distance(pointA , pointB): # PointA and pointB will be vector(1D Np array)
    return np.sqrt(np.sum((pointA - pointB)**2))

def Predict(X_train,y_train,X_test,k): # 2D vector
    prediction = []
    for i in X_test:
        distance = [Eucliedean_Distance(i , j) for j in X_train]
        k_indices = np.argsort(distance)[:k]
        categories = [y_train[j] for j in k_indices]
        prediction.append(Counter(categories).most_common(1)[0][0])
    return prediction

def Accuracy(expected , actual):
    acc = accuracy_score(expected , actual)
    print(f"Accuracy is {acc*100}%")

def Feature_Scaling(X_train , X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return (scaler.transform(X_train) , scaler.transform(X_test))

#######################################################################################

# Input Data 

df = pd.read_csv('gene_expression.csv')
X_train , X_test , y_train ,y_test = train_test_split(df.drop(columns = 'Cancer Present'),df['Cancer Present'],test_size=0.3)

# Lets remove the Feature Name from Train and test data 
X_train = X_train.values
X_test = X_test.values
y_test = y_test.values
y_train = y_train.values

# Lets Scale the features 
X_train , X_test = Feature_Scaling(X_train , X_test)


prediction = Predict(X_train , y_train ,X_test , 5)
Accuracy(y_test , prediction)

################################################################################################