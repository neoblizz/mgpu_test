#include <cstdlib>
#include <iostream>
#include <thrust/host_vector.h>
#include "memory.hxx"

#define PI 3.14159265358979323846  /* pi */

__global__ void kernel(float* x, std::size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        x[tid] = tid * PI;

        if(tid < 32)
            printf("%f\n", x[tid]);
    }
}

int main(int argc, char** argv) {
    const std::size_t N = 1<<30; // 1B
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    std::vector<int> devices;
    for(int i = 0; i < num_gpus; i++)
        devices.push_back(i);

    physical_memory<float> pm(N, devices); // Create physical memory for all devices.
    virtual_memory<float> vm(pm.padded_size);
    memory_mapper<float> map(vm, pm, devices);

    cudaSetDevice(0);
    int block_size = 256;
    int grid_size = (pm.padded_size + block_size - 1) / block_size;
    
    for (int iter = 0; iter < 100; iter++)
        kernel<<<grid_size, block_size>>>(map.data(), N);
    
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}