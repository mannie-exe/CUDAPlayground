#include <stdio.h>
#include <math.h>
#include <time.h>

/**
 *
 * Example/test adding function. Requires
 * length (n), and adds everything to
 * float *y (does NOT create new array!).
 *
 */
void test_add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    /**
    *
    * Setup a timer to calculate time spent
    * working at the end.
    *
    */
    time_t seconds = time(NULL);

    const int ARRAY_LENGTH = 1 << 26;
    const int ARRAY_MEM_SIZE = sizeof(float) * ARRAY_LENGTH;

    /**
    *
    * Allocate enough memory for two arrays of
    * size ARRAY_MEM_SIZE.
    *
    */
    float *array_x = (float *)malloc(ARRAY_MEM_SIZE);
    float *array_y = (float *)malloc(ARRAY_MEM_SIZE);
    if (!array_x || !array_y)
    {
        printf("what da fak, there's no memory?! there's no memory for %d bytes?! aborting...", ARRAY_MEM_SIZE * 2);
        return -1;
    }

    /**
    *
    * Fill both arrays with some test data, so
    * that adding each array at the same index
    * gives the float, 3.0F.
    *
    */
    for (int idx = 0; idx < ARRAY_LENGTH; idx++)
    {
        array_x[idx] = 1.0f;
        array_y[idx] = 2.0f;
    }
    test_add(ARRAY_LENGTH, array_x, array_y);

    /**
    *
    * Confirm each element of array_y to be 3.0F,
    * otherwise calculate out the margin-of-error.
    *
    */
    float maxError = 0.0F;
    for (int idx = 0; idx < ARRAY_LENGTH; idx++)
        maxError = fmax(maxError, fabs(array_y[idx] - 3.0F));

    free(array_x);
    free(array_y);

    printf("Time elpased: %lld; max float computation error: %f\n", time(NULL) - seconds, maxError);
    return 0;
}
