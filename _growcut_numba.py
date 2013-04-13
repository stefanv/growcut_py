from __future__ import division
import sys
import math
import numba
from numba import jit, autojit, size_t
import numpy as np
import numpy.testing as npt

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

def kernel(image, state, state_next, window_size):
    changes = 0

    height  = image.shape[0]
    width   = image.shape[1]

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
                        changes += 1

            state_next[i,j,0] = winning_colony
            state_next[i,j,1] = defense_strength

    return changes

def create_numba_funcs(scalar_type=SCALAR_TYPE):
    this = sys.modules[__name__]

    pixel_type = scalar_type[:]
    image_type = scalar_type[:,:,:]
    state_type = scalar_type[:,:,:]

    this.__numba_window_floor  = jit(inline=True,nopython=True,
                                     argtypes=[size_t,size_t],
                                     restype=size_t)(__py_window_floor)

    this.__numba_window_ceil   = jit(inline=True,nopython=True,
                                     argtypes=[size_t,size_t,size_t],
                                     restype=size_t)(__py_window_ceil)

    this.__numba_distance      = jit(inline=True,nopython=False,
                                     argtypes=[pixel_type,pixel_type],
                                     s=scalar_type,
                                     d=scalar_type,
                                     restype=scalar_type)(__py_distance)

    this.__numba_np_distance   = jit(inline=True,nopython=False,
                                     argtypes=[pixel_type,pixel_type],
                                     restype=scalar_type)(__py_np_distance)

    this.__numba_g             = jit(inline=True,nopython=False,
                                     argtypes=[pixel_type,pixel_type],
                                     restype=scalar_type)(__py_g)

    this.__numba_np_g          = jit(inline=True,nopython=False,
                                     argtypes=[pixel_type,pixel_type],
                                     restype=scalar_type)(__py_np_g)

    this.__numba_kernel = autojit(inline=True,nopython=False)(__py_kernel)
    # the below code does not work
    # this.__numba_kernel        = jit(nopython=False,
    #                                  argtypes=[image_type,
    #                                            state_type,
    #                                            state_type,
    #                                            size_t],
    #                                  restype=int_,
    #                                  attack_strength=scalar_type,
    #                                  defense_strength=scalar_type,
    #                                  winning_colony=scalar_type)(__py_kernel)


def debug():
    this = sys.modules[__name__]
    this.window_floor = __py_window_floor
    this.window_ceil  = __py_window_ceil
    this.distance     = __py_distance
    this.np_distance  = __py_np_distance
    this.g            = __py_g
    this.np_g         = __py_np_g
    this.kernel       = __py_kernel

def optimize():
    this = sys.modules[__name__]
    this.window_floor = __numba_window_floor
    this.window_ceil  = __numba_window_ceil
    this.distance     = __numba_distance
    this.np_distance  = __numba_np_distance
    this.g            = __numba_g
    this.np_g         = __numba_np_g
    this.kernel       = __numba_kernel

# protected Pythonic versions of code:
__py_window_floor  = window_floor
__py_window_ceil   = window_ceil
__py_distance      = distance
__py_np_distance   = np_distance
__py_g             = g
__py_np_g          = np_g
__py_kernel        = kernel

def test_window_floor_ceil():

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

    assert 1.0 == g(pixel1,pixel1)
    assert abs(0 - g(pixel1,pixel2)) < 1e-15
    assert abs(0.5 - g(pixel2,pixel3)) < 1e-15

    assert 1.0 == np_g(pixel1,pixel1)
    assert abs(0 - np_g(pixel1,pixel2)) < 1e-15
    assert abs(0.5 - np_g(pixel2,pixel3)) < 1e-15

def test_kernel():
    image = np.zeros((3,3,3), dtype=SCALAR_DTYPE)
    state = np.zeros((3,3,2), dtype=SCALAR_DTYPE)
    state_next = np.empty_like(state)

    # colony 1 is strength 1 at position 0,0
    # colony 0 is strength 0 at all other positions
    state[0,0,0] = 1
    state[0,0,1] = 1

    # window_size 1, colony 1 should propagate to three neighbors
    changes = kernel(image, state, state_next, 1)
    assert(3==changes)
    npt.assert_array_equal(state_next[0:2,0:2],1)
    npt.assert_array_equal(state_next[2,:],0)
    npt.assert_array_equal(state_next[2,:],0)

    # window_size 1, colony 1 should propagate to entire image
    changes = kernel(image, state, state_next, 2)
    assert(8==changes)
    npt.assert_array_equal(state_next,1)

def test():
    test_window_floor_ceil()
    test_distance()
    test_g()
    test_kernel()

# create numba versions of code
create_numba_funcs()

# replace default function calls with numba calls
optimize()

if __name__ == "__main__":
    test()
    debug()
    test()
