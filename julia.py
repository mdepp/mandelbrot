import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from matplotlib import pyplot

class JuliaSetRenderer:
    def __init__(self, npixels_x, npixels_y, max_iterations):
        """
        Initialize the renderer
        """
        self.npixels_x = npixels_x
        self.npixels_y = npixels_y
        self.max_iterations = max_iterations

        # Initialize OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        self.cl_res = cl_array.zeros(self.queue, (self.npixels_x*self.npixels_y,), np.float32)

        # Set up program kernel
        with open('julia.cl') as source:
            try:
                self.program = cl.Program(self.context, source.read()).build()
            except cl.cffi_cl.RuntimeError as err:
                raise RuntimeError('Could not compile program: {0}'.format(err))
            self.kernel = self.program.main
        if not self.kernel:
            raise RuntimeError('Could not load program kernel (does file exist?)')
        
        self.kernel.set_scalar_arg_dtypes([None, np.float32, np.int32])

    def get_image(self, time):
        """
        Get a view into the image at a certain time. Pixel data from this image
        is from 0 (not in the set) to max_iterations (in the set).
        """
        # Calculate image at this time using the kernel
        self.kernel(self.queue, (self.npixels_x, self.npixels_y), None, self.cl_res.data, time, self.max_iterations)
        # Get the result in a 2D view
        return self.cl_res.get().reshape((self.npixels_y, self.npixels_x))
    
    def plot_image(self, image):
        """
        Plot a generated image
        """
        pyplot.imshow(image, cmap='viridis')
        pyplot.show()


import imageio
from tqdm import tqdm
from matplotlib.pyplot import cm

def main():
    max_iterations = 100
    n_frames = 500
    fps = 60
    renderer = JuliaSetRenderer(640, int(640*2.0/3), max_iterations)
    time_axis = np.linspace(0, np.pi*2, n_frames)
    with imageio.get_writer('out.gif', mode='I', format='GIF', duration=1.0/fps) as writer:
        for time in tqdm(time_axis):
            image = renderer.get_image(time)
            writer.append_data(np.uint8(cm.gist_earth(image/max_iterations)*255.0))

main()