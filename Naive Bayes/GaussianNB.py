class GaussianNB:
    def __init__(self):
        self.prior_prob = {}
        self.mean = {}
        self.var = {}
        self.classes = []

    def fit(self,X,y):
        self.classes = np.unique(y)
        for cls in self.classes:
            class_data = X[y == cls]
            self.mean[cls] = np.mean(class_data,axis=0)
            self.var[cls] = np.var(class_data,axis=0)
            self.prior_prob[cls] = class_data.shape[0] / X.shape[0]

    def predict(self,X_test):
        result = [self.predict_single(x) for x in X_test]
        return np.array(result)
    
    def predict_single(self,x):
        posteriors = []
        for cls in self.classes:
            log_prior = np.log(self.prior_prob[cls])
            log_likelihood = np.sum(np.log(self.pdf(cls,x)))
            posterior = log_likelihood + log_prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def pdf(self,cls,x):
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = (2 * np.pi * var) ** 0.5
        return numerator * denominator   

        