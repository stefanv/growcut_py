from growcut_py import growcut
from _growcut_numba import growcut as growcut_numba
from _growcut import growcut as growcut_cython

import numpy as np
import numpy.testing as npt

from skimage import draw

from time import time

def setup():
    image = np.zeros((30, 30, 3), dtype=float)
    image[draw.ellipse(15, 15, 5, 8)] = 1

    state = np.zeros((30, 30, 2))
    state[15, 15] = (1, 1)
    state[0, 0] = (0, 1)

    return image, state

def test_growcut_basic():
    image, state = setup()
    t1 = time()
    out = growcut(image, state, window_size=3, max_iter=20)
    print("python: %e s" % (time()-t1))

    image,state = setup()
    t1 = time()
    out = growcut_cython(image, state, window_size=3, max_iter=20)
    print("cython: %e s" % (time()-t1))

    # let numba compile the function
    image,state = setup()
    out = growcut_numba(image, state, 20, 3)

    image,state = setup()
    t1 = time()
    out = growcut_numba(image, state, window_size=3, max_iter=20)
    print("numba: %e s" % (time()-t1))

    # image,state = setup()
    # npt.assert_array_equal(image[..., 0],
    #                        growcut(image, state, window_size=3, max_iter=20))
    # image,state = setup()
    # npt.assert_array_equal(image[..., 0],
    #                        growcut_cython(image, state, window_size=3, max_iter=20))
    # image,state = setup()
    # npt.assert_array_almost_equal(image[..., 0],
    #                        growcut_numba(image, state, window_size=3, max_iter=20))

if __name__ == "__main__":
    test_growcut_basic()
