import numpy as np

# Crate a numpy array of [10,2] 
a = np.zeros((10,2))
a[:,0] = 1
a[:,1] = 2


# Flatten it using reshape
b = a.reshape((20,1))
print(b)