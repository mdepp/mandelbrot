/*
 * This kernel produces images of an interesting-looking Julia set.
 */

kernel void main(global float* res, float t, int max_iterations)
{
    // Set program constants
    const float down = -2.0, left = -4.0, width = 6.0, height = 4.0;
    
    // Retrieve coordinates of current pixel
    const int2 pixel = (int2)(get_global_id(0), get_global_id(1));
    
    // Calculate the corresponding point on the complex plane
    const int2 npixels = (int2)(get_global_size(0), get_global_size(1));
    const float2 coord = (float2)(left + width*pixel.x/npixels.x, down + height*pixel.y/npixels.y);

    /*
     * As in https://en.wikipedia.org/wiki/Julia_set#Quadratic_polynomials,
     * 
     * Let t be real, and let f : Ĉ -> Ĉ,  z |--> z^2 + 0.7785exp(it). Then,
     *
     *     J(f) = {z in C : |(f^n)(z)| <= 2 for all n=0,1,2,...}
     *
     * So, to check if any complex number z is in J(f), we iterate through the
     * sequence (f^n)(z) for n until |(f^n)(z)|>=2 (or a maximum number of
     * iterations has been reached).
     */
    int iterations = 0;
    float2 val = coord; // Initial value of sequence
    float2 next_val;
    while (iterations < max_iterations && (val.x*val.x+val.y*val.y) < 4)
    {
        // Calculate next value in sequence from previous value
        next_val.x = val.x*val.x-val.y*val.y + 0.7785*cos(t);
        next_val.y = 2*val.x*val.y + 0.7785*sin(t);
        
        val = next_val;

        ++iterations;
    }
    
    res[pixel.y*npixels.x + pixel.x] = (float)iterations;
}