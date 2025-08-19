# Regularized Least Square for MyoControl
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
import numpy as np
from choldate import cholupdate

class rls_classifier():
    def __init__(
        self,
        lamb = 0.5,  # hyperparameter [0,1]
        nFeatures = 64,
        X = np.array([]),
        y = np.array([])
    ):
        self.lamb = lamb
        self.nFeatures = nFeatures
        
        if X.shape[0] == 0 or y.shape[0] == 0 :
            self.T = 0
            self.T_seen = {}
            self.gamma = np.zeros(self.T)
            self.R = np.sqrt(lamb) * np.eye(self.nFeatures)
            self.b = np.zeros((self.nFeatures, self.T))
            self.Wk = np.zeros((self.nFeatures, self.T))
            
        else:
            self.T = len(np.unique(y)) #Count how many unique classes are in y
            self.T_seen = {i:y.tolist().count(i) for i in y.tolist()} # How many samples have been already trained for each class
            self.gamma = list(self.T_seen.values()) # Initialized on the basis of frequency (info in T_seen)
            self.R = cholesky(np.matmul(X.T,X) + self.lamb * np.eye(self.nFeatures))
            Y_OneHot = np.zeros((len(y),self.T))
            for idx,val in enumerate(y):Y_OneHot[idx] = 1*(list(self.T_seen.keys()) == y[idx])
            self.b = np.matmul(X.T,Y_OneHot)
            Ud = solve_triangular(self.R, self.b, trans = 1, lower = False)
            self.Wk = solve_triangular(self.R, Ud, trans = 0, lower = False)      

    def learn_one(self, x, y):
        x = x.reshape(1,x.shape[0]).astype('float64')
        if y not in self.T_seen.keys():
            # Differentiate if new class is coming
            self.T += 1
            self.gamma = np.append(self.gamma,0)
            self.b = np.append(self.b,np.zeros((self.nFeatures,1)), axis = 1)
            self.T_seen[y] = 1
        else:
            self.T_seen[y] += 1
        # Update the b-value
        self.e = 1*(list(self.T_seen.keys()) == y)
        self.gamma = self.gamma + self.e # Gamma should contain the frequency of each already trained class at each iteration
        self.b = self.b + x.T * self.e
        cholupdate(self.R,x[0].copy())
        Ud = solve_triangular(self.R, self.b, trans = 1, lower = False)
        self.Wk = solve_triangular(self.R, Ud, trans = 0, lower = False)    

        pass
    
    def learn_many(self, X, Y):
        # TODO: Implement later
        pass

    def predict_one(self, x_new):
        idx_y_pred = np.argmax(np.matmul(self.Wk.T,x_new),axis = 0)
        list_T_seen = list(self.T_seen)
        y_pred = list_T_seen[idx_y_pred]
        return y_pred

    