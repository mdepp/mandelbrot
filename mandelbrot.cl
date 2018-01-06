 __kernel void mandelbrot(__global float* res)
{
    // Set program constants (these should really be set from the main program
    // somehow...
    const float down = -1.0, left = -2.0, width = 3.0, height = 2.0;
    const int npixels_x = 1920*3, npixels_y = 1920*2;
    const int max_iterations = 1000;
    
    const int gid = get_global_id(0);
    // Calculate coordinates of the current pixel
    const float pixel_x = gid % npixels_x;
    const float pixel_y = gid / npixels_x;
    // Calculate the corresponding point on the complex plane
    const float coord_re = left + width*pixel_x/npixels_x;
    const float coord_im = down + height*pixel_y/npixels_y;

    // Init real and imaginary parts of sequence to 0
    float re = 0, im = 0;
    float new_re, new_im;
    
    //bool is_convergent = true;
    int i;
    for (i=0; i<max_iterations; ++i)
    {
        // Sequence is defined recursively as a_{n+1} = a_n + coord
        new_re = re*re-im*im + coord_re;
        new_im = 2*re*im + coord_im;
        // If current sequence value is >= 2 along real or imaginary axis,
        // sequence is growing arbitrarily large, so by definition the current
        // complex coordinate is not in the Mandelbrot set.
        if (new_re >= 2 || new_im >= 2)
        {
            //is_convergent = false;
            break;
        }
        re = new_re;
        im = new_im;
    }
    // If by max_iterations iteration the sequence is not growing arbitrarily large,
    // it is assumed that it will never do so, so the current coordinate is indeed
    // in the Mandelbrot set.
    
    // Returning this instead provides a nice visual effect and helps make clear
    // features of the set with fewer iterations.
    res[gid] = (float)i;
    //return is_convergent ? 0.0 : 1.0;
}