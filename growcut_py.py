from __future__ import division

import numpy as np
from skimage import io, img_as_float
from math import sqrt


def g(x, y):
    return 1 - np.sqrt(np.sum((x - y) ** 2)) / sqrt(3)


def growcut(image, state, max_iter=500, window_size=5):
    """Grow-cut segmentation.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    state : (M, N, 2) ndarray
        Initial state, which stores (foreground/background, strength) for
        each pixel position or automaton.  The strength represents the
        certainty of the state (e.g., 1 is a hard seed value that remains
        constant throughout segmentation).
    max_iter : int, optional
        The maximum number of automata iterations to allow.  The segmentation
        may complete earlier if the state no longer varies.
    window_size : int
        Size of the neighborhood window.

    Returns
    -------
    mask : ndarray
        Segmented image.  A value of zero indicates background, one foreground.

    """
    image = img_as_float(image)
    height, width = image.shape[:2]
    ws = (window_size - 1) // 2

    changes = 1
    n = 0

    state_next = state.copy()

    while changes > 0 and n < max_iter:
        changes = 0
        n += 1

        print n

        for j in range(width):
            for i in range(height):
                C_p = image[i, j]
                S_p = state[i, j]
                changes_per_cell = 0

                for jj in xrange(max(0, j - ws), min(j + ws + 1, width)):
                    for ii in xrange(max(0, i - ws), min(i + ws + 1, height)):
                        if (ii == i and jj == j) or (changes_per_cell != 0):
                            continue

                        # p -> current cell
                        # q -> attacker
                        C_q = image[ii, jj]
                        S_q = state[ii, jj]

                        gc = g(C_q, C_p)

                        if gc * S_q[1] > S_p[1]:
                            state_next[i, j, 1] = gc * S_q[1]

                            if S_p[0] != S_q[0]:
                                state_next[i, j, 0] = S_q[0]
                                changes += 1
                                changes_per_cell += 1
                                break

        state = state_next

    return state[:, :, 0]


if __name__ == "__main__":
    image = io.imread('sharkfin_small.jpg')
    state = np.zeros((image.shape[0], image.shape[1], 2))
    state[90, 90] = (1, 1)
    state[0, 0] = (0, 1)

    out = growcut(image, state, window_size=3, max_iter=20)

    import matplotlib.pyplot as plt

    f, (ax0, ax1) = plt.subplots(1, 2)

    ax0.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax0.set_title('Input image')

    ax1.imshow(out, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    ax1.set_title('Foreground / background')

    plt.show()
