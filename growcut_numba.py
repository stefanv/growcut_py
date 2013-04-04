from __future__ import division

import numpy as np
from skimage import io, img_as_float
from math import sqrt

from numba import autojit, jit, float_, int_, size_t, struct

NUMBAFY = True

def size_t_min(x,y):
    if x > y: return y
    else: return x

def size_t_max(x,y):
    if x > y: return x
    else: return y

def g(x, y):
    return 1.0 - np.linalg.norm(x-y,2)/sqrt(3.0)
#    return 1.0 - np.sqrt(np.sum((x - y) ** 2)) / np.sqrt(3)

def distance(image, r0, c0, r1, c1):
    s = 0.0
    for i in range(3):
        d = image[r0, c0, i] - image[r1, c1, i]
        s += d * d
    return sqrt(s)

s3 = sqrt(3.0)

def g_cython(d):
    return 1.0 - (d / s3)

def growcut_inner(ws,width,height,image,state,state_next,changes):
    """changes, state_next = growcut_inner(ws,width,height,image,state,state_next,changes)"""
    for j in range(width):
        for i in range(height):
            changes_per_cell = 0
            for jj in xrange(size_t_max(0, j - ws), size_t_min(j + ws + 1, width)):
                for ii in xrange(size_t_max(0, i - ws), size_t_min(i + ws + 1, height)):
                    if (ii == i and jj == j) or (changes_per_cell != 0):
                        continue

                    #gc = g(image[i, j], image[ii, jj])
                    gc = g_cython(distance(image,i,j,ii,jj))

                    attack_strength = gc * state[ii, jj, 1]

                    if attack_strength > state[i, j, 1]:
                        state_next[i, j, 1] = attack_strength

                        if state[i, j, 0] != state[ii, jj, 0]:
                            state_next[i, j, 0] = state[ii, jj, 0]

                            changes += 1
                            changes_per_cell += 1
                            break
    return changes

if NUMBAFY:
    size_t_min    = jit(inline=True,nopython=True,
                        argtypes=[size_t,size_t],restype=size_t)(size_t_min)

    size_t_max    = jit(inline=True,nopython=True,
                        argtypes=[size_t,size_t],restype=size_t)(size_t_max)

    g             = jit(inline=True,nopython=False,
                        argtypes=[float_[:], float_[:]],
                        restype=float_)(g)

    distance      = jit(inline=True,nopython=True,
                        argtypes=[float_[:,:,:],size_t,size_t,size_t,size_t],
                        restype=float_)(distance)

    g_cython      = jit(inline=True,nopython=True,
                        argtypes=[float_],restype=float_)(g_cython)

    growcut_inner = jit(inline=True,nopython=True,
                        argtypes=[size_t,size_t,size_t,
                                  float_[:,:,:],float_[:,:,:],float_[:,:,:],
                                  int_])(growcut_inner)

def growcut_numba(image, state, max_iter=20, window_size=3):
    """Grow-cut segmentation (Numba acceleration).

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
    window_size : int, optional
        Size of the neighborhood window.

    Returns
    -------
    mask : ndarray
        Segmented image.  A value of zero indicates background, one foreground.

    """
    #image = img_as_float(image)
    #height, width = image.shape[:2]
    height = image.shape[0]
    width = image.shape[1]

    ws = (window_size - 1) // 2

    changes = 1
    n = 0

    state_next = state.copy()

    while changes > 0 and n < max_iter:
        changes = 0
        n += 1
        changes = growcut_inner(ws,width,height,image,state,state_next,changes)

        swap_state = state_next
        state_next = state
        state      = swap_state
        print n, changes

    return state_next[:, :, 0]

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
