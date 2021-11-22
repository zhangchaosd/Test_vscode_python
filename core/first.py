import os
import sys
import numpy as np
sys.path.append('.')
from utils.addd import addd
def ns(data):
    return normalization(standardization(data))

def normalization( data):
    _range = np.max(abs(data))
    return data / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
# -b 8
if __name__ == "__main__":
#     print(os.getcwd())
    print(addd(12,9))
#     print(sys.path)


    sz1 = [1., 2.]
    sz2 = [1.,2.,3.]
    sz3 = ns(sz1)
    sz4 = ns(sz2)
    print(sz1 + sz2)
    print(sz3.tolist()+sz4.tolist())