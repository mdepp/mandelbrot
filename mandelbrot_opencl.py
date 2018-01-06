import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from matplotlib import pyplot

# Init constants (the rest of these are set from the kernel)
npixels_x = 1920*3*2
npixels_y = 1920*2*2

# Init CL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Set up an array to store the calculation results
cl_res = cl_array.zeros(queue, (npixels_x*npixels_y,), np.float32)

with open('mandelbrot.cl') as f:
    # Build and run program
    program = cl.Program(context, f.read()).build()
    mandelbrot = program.mandelbrot
    mandelbrot(queue, (npixels_x, npixels_y), None, cl_res.data)

    # Get the result in a 2D view for rendering
    pixels = cl_res.get().reshape((npixels_y, npixels_x))
    
    # Display result
    pyplot.matshow(pixels, cmap='viridis')
    pyplot.show()