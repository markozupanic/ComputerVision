import numpy as np
import random


numpy_polje=np.random.randint(100,size=(12,15),dtype=np.uint16)

novo_numpy_polje=numpy_polje[::,0::2]



print(numpy_polje)
print(numpy_polje.dtype)
print(novo_numpy_polje)
print(novo_numpy_polje.shape)






