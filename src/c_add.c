#include <stdio.h>
#include <math.h>
#include <time.h>

float *create_host_float_array(const int LENGTH)
{
    const int MEM_SIZE = sizeof(float) * LENGTH;
    float *array = (float *)malloc(MEM_SIZE);
    if (!array)
    {
        fprintf(stderr, "Failed to allocate array of floats on the host (CPU) with length %d and size %d; exiting...\n", LENGTH, MEM_SIZE);
        exit(-420);
    }
    return array;
}

/**
*
* Adds two float arrays into a third array.
* All arrays must of the same size of variable
* LENGTH.
*
*/
void add_floats_of_length(float *dest, float *src_a, float *src_b, const int LENGTH)
{
    for (int idx = 0; idx < LENGTH; idx++)
    {
        dest[idx] = src_a[idx] + src_b[idx];
        fprintf(stdout, "%f + %f = %f", src_a[idx], src_b[idx], dest[idx]);
    }
}

int main(void)
{
    /**
    *
    * Setup a timer to calculate time spent
    * working at the end.
    *
    */
    time_t start_time = time(NULL);

    const int ARRAY_LENGTH = 1 << 26;
    float *input_a = create_host_float_array(ARRAY_LENGTH);
    float *input_b = create_host_float_array(ARRAY_LENGTH);
    float *output = create_host_float_array(ARRAY_LENGTH);

    /**
    *
    * Fill both arrays with some test data, so
    * that adding each array at the same index
    * gives the float "3.0F"
    *
    */
    for (int idx = 0; idx < ARRAY_LENGTH; idx++)
    {
        input_a[idx] = 1.0f;
        input_b[idx] = 2.0f;
    }

    add_floats_of_length(output, input_a, input_b, ARRAY_LENGTH);

    /**
    *
    * Confirm each element of output to be 3.0F,
    * otherwise calculate out the margin-of-error.
    *
    */
    float maxError = 0.0F;
    for (int idx = 0; idx < ARRAY_LENGTH; idx++)
        maxError = fmax(maxError, fabs(output[idx] - (double)3.0F));

    fprintf(stdout, "--- END ---\nTime elapsed: %llds\nMax float computation error: %f\n--- END ---\n", time(NULL) - start_time, maxError);

    free(input_a);
    free(input_b);
    free(output);

    return 0;
}
