from _growcut import growcut
import numpy as np
from skimage import io

image = io.imread('sharkfin_small.jpg')
state = np.zeros((image.shape[0], image.shape[1], 2))

foreground_pixels = np.array([(90, 90), (50, 90)])
background_pixels = np.array([(5, 5), (5, 145), (120, 10), (60, 20), (10, 110), (100, 160)])

for (r, c) in background_pixels:
    state[r, c] = (0, 1)

for (r, c) in foreground_pixels:
    state[r, c] = (1, 1)

out = growcut(image, state, window_size=5, max_iter=500)

import matplotlib.pyplot as plt

f, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
ax0.plot(foreground_pixels[:, 1], foreground_pixels[:, 0],
         color='blue', marker='o', linestyle='none', label='Foreground')
ax0.plot(background_pixels[:, 1], background_pixels[:, 0],
         color='red', marker='o', linestyle='none', label='Background')
ax0.set_title('Input image')
ax0.axis('image')

ax1.imshow(out, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
ax1.set_title('Foreground / background')
ax0.axis('image')

plt.show()
