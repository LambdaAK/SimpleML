import numpy as np
import time

# time how long it takes to compute the inverse of the n by n identity matrix for n = 1, ... 500

for n in range(3000, 5001):
    start = time.time()
    np.linalg.inv(np.eye(n))
    end = time.time()
    print(end - start)
