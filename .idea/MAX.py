__author__ = 'mikhail'
import numpy as np
from skimage import filter


arr = np.array([[4, 4, 4, 4, 4, 4, 4, 4],
    [3, 2, 1, 2, 1, 2, 1, 3],
    [3, 1, 2, 1, 2, 1, 2, 3],
    [3, 2, 1, 2, 1, 2, 1, 3],
    [3, 1, 2, 1, 2, 1, 2, 3],
    [3, 2, 1, 2, 1, 2, 1, 3],
    [3, 1, 2, 1, 2, 1, 2, 3],
    [4, 4, 4, 4, 4, 4, 4, 4]])

gaussian = filter.gaussian_filter(arr)
roberts = filter.roberts(arr)

print gaussian
print roberts