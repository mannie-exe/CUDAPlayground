#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

float *create_host_float_array(const int LENGTH)
{
    const int MEM_SIZE = sizeof(float) * LENGTH;
    float *array = (float *)malloc(MEM_SIZE);
    if (!array)
    {
        fprintf(stderr, "Failed to allocate array of floats on the host (CPU) with length %d and size %d; exiting...\n", LENGTH, MEM_SIZE);
        exit(EXIT_FAILURE);

    }
    return array;
}

float *create_device_float_array(const int LENGTH)
{
    const int MEM_SIZE = sizeof(float) * LENGTH;
    float *array = NULL;
    cudaError_t err = cudaMalloc((void **)&array, MEM_SIZE);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate array of floats on the CUDA-device with length %d and size %d; CUDA error code: %s; exiting...\n", LENGTH, MEM_SIZE, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return array;
}

void copy_mem_host_to_device(void *host_pointer, void *device_pointer, const int MEM_SIZE)
{
    cudaError_t err = cudaMemcpy(device_pointer, host_pointer, MEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy host pointer (%p) into device pointer (%p) with size %d; exiting...\n", host_pointer, device_pointer, MEM_SIZE, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void copy_mem_device_to_host(void *device_pointer, void *host_pointer, const int MEM_SIZE)
{
    cudaError_t err = cudaMemcpy(host_pointer, device_pointer, MEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy device pointer (%p) into host pointer (%p) with size %d; exiting...\n", device_pointer, host_pointer, MEM_SIZE, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void delete_device_memory(void *memory)
{
    cudaError_t err = cudaFree(memory);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free memory from device; CUDA error code: %s; exiting...\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
*
* Example "kernel" function (runs on the GPU).
*
* Adds two float arrays into a third array.
* All arrays must of the same size of variable
* LENGTH.
*
* Note: __global__ makes this a kernel function.
*
*/
__global__
void add_floats_of_length(float *input_a, float *input_b, float *output, const int LENGTH)
{
    const int block_offset = blockDim.x * gridDim.x;
    const int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = thread_offset; idx < LENGTH; idx += block_offset)
        output[idx] = input_a[idx] + input_b[idx];
}

int main(void)
{
    /**
    *
    * Setup a timer to calculate time spent
    * working at the end.
    *
    */
    time_t start = time(NULL);

    /**
    *
    * Length and memory sizeof all input/output,
    * and host/device arrays.
    *
    */
    time_t start_mem_alloc = time(NULL);
    const int ARRAY_LENGTH = 1 << 26;
    const int ARRAY_MEM_SIZE = sizeof(float) * ARRAY_LENGTH;
    // Initialize host (CPU) arrays
    float *host_input_a = create_host_float_array(ARRAY_LENGTH);
    float *host_input_b = create_host_float_array(ARRAY_LENGTH);
    float *host_output = create_host_float_array(ARRAY_LENGTH);
    // Initialize CUDA device arrays
    float *device_input_a = create_device_float_array(ARRAY_LENGTH);
    float *device_input_b = create_device_float_array(ARRAY_LENGTH);
    float *device_output = create_device_float_array(ARRAY_LENGTH);
    fprintf(stdout, "Took %llds to allocate %d bytes of memory\n", time(NULL) - start_mem_alloc, ARRAY_MEM_SIZE * 6);

    /**
    *
    * Fill both arrays with some test data, so
    * that adding each array at the same index
    * gives the float, 3.0F.
    *
    */
    time_t start_array_fill = time(NULL);
    for (int idx = 0; idx < ARRAY_LENGTH; idx++)
    {
        host_input_a[idx] = 1.0f;
        host_input_b[idx] = 2.0f;
    }
    fprintf(stdout, "Took %llds to fill host input arrays\n", time(NULL) - start_array_fill);

    /**
    *
    * Copy host input arrays into device memory.
    *
    */
    time_t start_mem_copy_h2d = time(NULL);
    copy_mem_host_to_device(host_input_a, device_input_a, ARRAY_MEM_SIZE);
    copy_mem_host_to_device(host_input_b, device_input_b, ARRAY_MEM_SIZE);
    fprintf(stdout, "Took %llds to copy host inputs into device\n", time(NULL) - start_mem_copy_h2d);

    /**
    *
    * Run a basic addition test on the device
    * with the copied (and now available) inputs.
    *
    */
    time_t start_device_func = time(NULL);
    const int block_size = 1024;
    const int grid_size = (ARRAY_LENGTH + block_size - 1) / block_size;
    add_floats_of_length<<<grid_size, block_size>>>(device_input_a, device_input_b, device_output, ARRAY_LENGTH);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to perform device calculation; CUDA error code: %s; exiting...\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    fprintf(stdout, "Took %llds to perform %d operations [%d][%d]\n", time(NULL) - start_device_func, grid_size * block_size, grid_size, block_size);

    /**
    *
    * Copy device output to host output.
    *
    */
    time_t start_mem_copy_d2h = time(NULL);
    copy_mem_device_to_host(device_output, host_output, ARRAY_MEM_SIZE);
    fprintf(stdout, "Took %llds to copy device output into host\n", time(NULL) - start_mem_copy_d2h);

    /**
    *
    * Confirm each element of array_y to be 3.0F,
    * otherwise calculate out the margin-of-error.
    *
    */
    float maxError = 0.0F;
    for (int idx = 0; idx < ARRAY_LENGTH; idx++)
        maxError = fmax(maxError, fabs(host_output[idx] - 3.0F));

    printf("--- END ---\nTotal time elpased: %llds\nMax float computation error: %f\n--- END ---\n", time(NULL) - start, maxError);

    delete_device_memory(device_input_a);
    delete_device_memory(device_input_b);
    delete_device_memory(device_output);
    free(host_input_a);
    free(host_input_b);
    free(host_output);

    return 0;
}
