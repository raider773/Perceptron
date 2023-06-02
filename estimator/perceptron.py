import numpy as np
from tqdm import tqdm

class Perceptron():
    def __init__(self,input_shape,transfer_function = "heaviside"):
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=input_shape)
        self.transfer_function  = transfer_function
    
    def transfer(self,x):
        if self.transfer_function.lower() == "heaviside":           
            return np.where(x >= 0, 1, 0)
            
        elif self.transfer_function.lower() == "sgn":
            return np.where(x == 0, 0, np.where(x > 0,1,-1))       
    
    def foward(self,x):
        return self.transfer(np.dot(x,self.weights))
    
    def Widrow_Hoff_rule(self,x,y,lr = 0.001,epochs = 10):        
        for i in tqdm(range(epochs)):
            for sample_index in range(x.shape[0]):     
                error =  y[sample_index] - self.foward(x[sample_index])
                self.weights +=  (error * x[sample_index] * lr)   