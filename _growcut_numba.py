from __future__ import division
import sys
import math
import numba
from numba import jit, autojit, size_t
import numpy as np
import numpy.testing as npt

NUMBAFY = True
SCALAR_DTYPE = np.float64
# This doesn't work :(
# SCALAR_TYPE  = numba.typeof(SCALAR_DTYPE)
SCALAR_TYPE = numba.float64

def window_floor(idx,size):
    if size > idx: return 0
    else         : return idx - size

def window_ceil(idx,ceil,size):
    if idx + size > ceil: return ceil
    else                : return idx + size

def distance(pixel1, pixel2):
    d = pixel1[0] - pixel2[0]
    s = d*d
    for i in range(1,3):
        d = pixel1[i] - pixel2[i]
        s += d*d
    return math.sqrt(s)

def np_distance(pixel1, pixel2):
    return np.linalg.norm(pixel1-pixel2,2)

sqrt_3 = math.sqrt(3.0)
def g(x,y):
    return 1.0 - distance(x,y)/sqrt_3

def np_g(x,y):
    return 1.0 - np_distance(x,y)/sqrt_3

def kernel(image, width, height, state, state_next, window_size):
    for j in xrange(width):
        for i in xrange(height):

            winning_colony   = state[i,j,0]
            defense_strength = state[i,j,1]

            for jj in xrange(window_floor(j,window_size),
                             window_ceil(j+1,width,window_size)):
                for ii in xrange(window_floor(i,window_size),
                                 window_ceil(i+1,height,window_size)):

                    gval = g(image[i,j],image[ii,jj])
                    attack_strength = gval * state[ii,jj,1]

                    if attack_strength > defense_strength:
                        defense_strength = attack_strength
                        winning_colony = state[ii,jj,0]

            state_next[i,j,0] = winning_colony
            state_next[i,j,1] = defense_strength


def kernel_2(image, state, state_next, window_size):
    height, width, depth = image.shape

    for j in xrange(width):
        for i in xrange(height):

            winning_colony   = state[i,j,0]
            defense_strength = state[i,j,1]

            for jj in xrange(window_floor(j,window_size),
                             window_ceil(j+1,width,window_size)):
                for ii in xrange(window_floor(i,window_size),
                                 window_ceil(i+1,height,window_size)):

                    gval = g(image[i,j],image[ii,jj])
                    attack_strength = gval * state[ii,jj,1]

                    if attack_strength > defense_strength:
                        defense_strength = attack_strength
                        winning_colony = state[ii,jj,0]

            state_next[i,j,0] = winning_colony
            state_next[i,j,1] = defense_strength


# protected Pythonic versions of code:
__window_floor  = window_floor
__window_ceil   = window_ceil
__distance      = distance
__np_distance   = np_distance
__g             = g
__np_g          = np_g

def numbafy(pixel_type=SCALAR_TYPE[:], pixel_scalar_type=SCALAR_TYPE):
    this = sys.modules[__name__]

    this.window_floor  = jit(inline=True,nopython=True,
                             argtypes=[size_t,size_t],
                             restype=size_t)(__window_floor)

    this.window_ceil   = jit(inline=True,nopython=True,
                             argtypes=[size_t,size_t,size_t],
                             restype=size_t)(__window_ceil)

    this.distance      = jit(inline=True,nopython=False,
                             argtypes=[pixel_type,pixel_type],
                             s=pixel_scalar_type,
                             d=pixel_scalar_type,
                             restype=pixel_scalar_type)(__distance)

    this.np_distance   = jit(inline=True,nopython=False,
                             argtypes=[pixel_type,pixel_type],
                             restype=pixel_scalar_type)(__np_distance)

    this.g             = jit(inline=True,nopython=False,
                             argtypes=[pixel_type,pixel_type],
                             restype=pixel_scalar_type)(__g)

    this.np_g          = jit(inline=True,nopython=False,
                             argtypes=[pixel_type,pixel_type],
                             restype=pixel_scalar_type)(__np_g)

if NUMBAFY:
    numbafy()

def test_window_floor_ceil():

    window_floor  = jit(inline=True,nopython=True,
                        argtypes=[size_t,size_t],
                        restype=size_t)(__window_floor)

    window_ceil  = jit(inline=True,nopython=True,
                       argtypes=[size_t,size_t,size_t],
                       restype=size_t)(__window_ceil)

    assert 3 == window_floor(4,1)
    assert 0 == window_floor(1,4)

    assert 3 == window_ceil(3,3,1)
    assert 5 == window_ceil(4,5,1)

def test_distance():
    pixel1 = np.asarray([0.0, 0.0, 0.0], dtype=SCALAR_DTYPE)
    pixel2 = np.asarray([1.0, 1.0, 1.0], dtype=SCALAR_DTYPE)
    pixel3 = np.asarray([0.5, 0.5, 0.5], dtype=SCALAR_DTYPE)
    pixel_scalar = np.asarray(0.0, dtype=SCALAR_DTYPE)

    pixel_type        = SCALAR_TYPE[:]
    pixel_scalar_type = SCALAR_TYPE

    distance = jit(inline=True,nopython=False,
                   argtypes=[pixel_type,pixel_type],
                   restype=pixel_scalar_type)(__distance)

    np_distance = jit(inline=True,nopython=False,
                   argtypes=[pixel_type,pixel_type],
                   restype=pixel_scalar_type)(__np_distance)

    assert 0.0 == distance(pixel1, pixel1)
    assert abs(math.sqrt(3) - distance(pixel1, pixel2)) < 1e-15
    assert abs(math.sqrt(3/4) - distance(pixel2, pixel3)) < 1e-15

    assert 0.0 == np_distance(pixel1, pixel1)
    assert abs(math.sqrt(3) - np_distance(pixel1, pixel2)) < 1e-15
    assert abs(math.sqrt(3/4) - np_distance(pixel2, pixel3)) < 1e-15

def test_g():
    pixel1 = np.asarray([0.0, 0.0, 0.0], dtype=SCALAR_DTYPE)
    pixel2 = np.asarray([1.0, 1.0, 1.0], dtype=SCALAR_DTYPE)
    pixel3 = np.asarray([0.5, 0.5, 0.5], dtype=SCALAR_DTYPE)

    pixel_type        = SCALAR_TYPE[:]
    pixel_scalar_type = SCALAR_TYPE

    g           = jit(inline=True,nopython=False,
                      argtypes=[pixel_type,pixel_type],
                      restype=pixel_scalar_type)(__g)

    np_g        = jit(inline=True,nopython=False,
                      argtypes=[pixel_type,pixel_type],
                      restype=pixel_scalar_type)(__np_g)

    assert 1.0 == g(pixel1,pixel1)
    assert abs(0 - g(pixel1,pixel2)) < 1e-15
    assert abs(0.5 - g(pixel2,pixel3)) < 1e-15

    assert 1.0 == np_g(pixel1,pixel1)
    assert abs(0 - np_g(pixel1,pixel2)) < 1e-15
    assert abs(0.5 - np_g(pixel2,pixel3)) < 1e-15

def test_kernel():
    image = np.zeros((3,3,2), dtype=SCALAR_DTYPE)
    state = np.zeros((3,3,2), dtype=SCALAR_DTYPE)

    # colony 1 is strength 1 at position 0,0
    # colony 0 is strength 0 at all other positions
    state[0,0,0] = 1
    state[0,0,1] = 1

    # window_size 1, colony 1 should propagate to three neighbors
    state_next = state.copy()
    kernel(image, 3, 3, state, state_next, 1)
    npt.assert_array_equal(state_next[0:2,0:2],1)
    npt.assert_array_equal(state_next[2,:],0)
    npt.assert_array_equal(state_next[2,:],0)

    # window_size 1, colony 1 should propagate to entire image
    state_next = state.copy()
    kernel(image, 3, 3, state, state_next, 2)
    npt.assert_array_equal(state_next,1)

if __name__ == "__main__":
    test_window_floor_ceil()
    test_distance()
    test_g()
    test_kernel()
