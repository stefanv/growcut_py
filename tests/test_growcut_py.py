from growcut_py import growcut
from growcut_py import growcut_fast

import numpy as np
import numpy.testing as npt

from skimage import draw


def test_growcut_basic():
    image = np.zeros((30, 30, 3), dtype=float)
    image[draw.ellipse(15, 15, 5, 8)] = 1

    state = np.zeros((30, 30, 2))
    state[15, 15] = (1, 1)
    state[0, 0] = (0, 1)

    npt.assert_array_equal(image[..., 0],
                           growcut(image, state, window_size=3, max_iter=20))

    npt.assert_array_equal(image[..., 0],
                           growcut_fast(image, state, window_size=3, max_iter=20))
