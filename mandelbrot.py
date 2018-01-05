import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool 

lower_left = -1-1.j
upper_right = 0+1.j

width = 250
height = 500
maximum_iterations = 500

in_grid = np.zeros(shape=(width,height), dtype=np.complex)

def getColour(args):
    """
    Returns the colour value of a single pixel, based on whether or not it is
    in the Mandelbrot set.
    """
    (x, y), val = args
    coord_x = x/width*np.real(upper_right-lower_left) + np.real(lower_left)
    coord_y = y/height*np.imag(upper_right-lower_left) + np.imag(lower_left)
    coord = coord_x + 1.j*coord_y

    is_convergent = True
    for i in range(maximum_iterations):
        in_grid[x,y] = np.square(in_grid[x,y]) + coord
        new_val = in_grid[x,y]
        if abs(np.real(new_val)) >= 2 or abs(np.imag(new_val)) >= 2:
            # Sequence diverges
            is_convergent = False
            break
    col = i
    return x, y, col


pool = ThreadPool()
out_grid = np.ndarray(shape=(width,height))
for x, y, col in tqdm(pool.imap_unordered(getColour, np.ndenumerate(in_grid)), total=width*height):
    out_grid[x,y] = col
pool.close()
pool.join()

out_grid = np.array(out_grid)
print(out_grid)

plt.matshow(out_grid.reshape((width, height)), cmap='viridis')
plt.show()