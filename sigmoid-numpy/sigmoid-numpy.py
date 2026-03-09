import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    
    x = np.array(x, dtype=float)

    return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
    

    