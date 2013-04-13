from __future__ import division
import sys
import math
import numba
from numba import jit, autojit, size_t
import numpy as np

NUMBAFY = False
SCALAR_DTYPE = np.float64
# This doesn't work :(
# SCALAR_TYPE  = numba.typeof(SCALAR_DTYPE)
SCALAR_TYPE = numba.float64

def size_t_min(x,y):
    if x > y: return y
    else: return x

def size_t_max(x,y):
    if x > y: return x
    else: return y

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

# protected Pythonic versions of code:
__size_t_min    = size_t_min
__size_t_max    = size_t_max
__distance      = distance
__np_distance   = np_distance
__g             = g
__np_g          = np_g

def numbafy(pixel_type=SCALAR_TYPE[:], pixel_scalar_type=SCALAR_TYPE):
    this = sys.modules[__name__]

    this.size_t_min  = jit(inline=True,nopython=True,
                           argtypes=[size_t,size_t],
                           restype=size_t)(__size_t_min)

    this.size_t_max  = jit(inline=True,nopython=True,
                           argtypes=[size_t,size_t],
                           restype=size_t)(__size_t_max)

    this.distance    = jit(inline=True,nopython=False,
                           argtypes=[pixel_type,pixel_type],
                           s=pixel_scalar_type,
                           d=pixel_scalar_type,
                           restype=pixel_scalar_type)(__distance)

    this.np_distance = jit(inline=True,nopython=False,
                           argtypes=[pixel_type,pixel_type],
                           restype=pixel_scalar_type)(__np_distance)

    this.g           = jit(inline=True,nopython=False,
                           argtypes=[pixel_type,pixel_type],
                           restype=pixel_scalar_type)(__g)

    this.np_g        = jit(inline=True,nopython=False,
                           argtypes=[pixel_type,pixel_type],
                           restype=pixel_scalar_type)(__np_g)

if NUMBAFY:
    numbafy()

def test_min_max():
    size_t_min = jit(inline=True,nopython=True,
                     argtypes=[size_t,size_t],restype=size_t)(__size_t_min)

    assert 3 == __size_t_min(3,4)
    assert 3 == __size_t_min(4,3)
    assert 0 == __size_t_min(0,0)

    assert 3 == size_t_min(3,4)
    assert 3 == size_t_min(4,3)
    assert 0 == size_t_min(0,0)

    size_t_max = jit(inline=True,nopython=True,
                     argtypes=[size_t,size_t],restype=size_t)(__size_t_max)

    assert 4 == __size_t_max(3,4)
    assert 4 == __size_t_max(4,3)
    assert 0 == __size_t_max(0,0)

    assert 4 == size_t_max(3,4)
    assert 4 == size_t_max(4,3)
    assert 0 == size_t_max(0,0)

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

if __name__ == "__main__":
    test_min_max()
    test_distance()
    test_g()
