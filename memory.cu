#include <cstdlib>
#include <iostream>
#include <thread>

#include <nvToolsExt.h>

#include <thrust/host_vector.h>
#include "memory.hxx"

struct my_timer_t {
  float time;

  my_timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~my_timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  // Alias of each other, start the timer.
  void begin() { cudaEventRecord(start_); }
  void start() { this->begin(); }

  float end() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);

    return milliseconds();
  }

  float seconds() { return time * 1e-3; }
  float milliseconds() { return time; }

 private:
  cudaEvent_t start_, stop_;
};

#define PI 3.14159265358979323846  /* pi */

__global__ void kernel(float* x, int offset, std::size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid + offset < N) {
        int device_id;
        cudaGetDevice(&device_id);

        x[tid] = tid * PI + device_id;
    
        // if(tid == 0)
        //     printf("device id = %u, offset = %u, elements_end = %u\n", device_id, offset, N);

        // if(tid < 32)
        //     printf("%f\n", x[tid]);

        // __syncthreads();
    }
}

int main(int argc, char** argv) {
    const std::size_t N = 1<<30; // 1B
    int num_gpus = 1;
    cudaGetDeviceCount(&num_gpus);

    std::cout << "Number of GPUs = " << num_gpus << std::endl;

    std::vector<int> devices;
    for(int i = 0; i < num_gpus; i++)
        devices.push_back(i);

    struct gpu_info {
        cudaStream_t stream;
        cudaEvent_t  event;
    };

    std::vector<gpu_info> infos;
    cudaStream_t master_stream;
  
    cudaSetDevice(0);
    cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);

    for(int i = 0 ; i < num_gpus ; i++) {
        gpu_info info;
        cudaSetDevice(i);
        cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&info.event, cudaEventDisableTiming);
        infos.push_back(info);
    }

    cudaSetDevice(0);

    std::size_t size = N * sizeof(float);
    physical_memory<float> pm(size, devices); // Create physical memory for all devices.
    virtual_memory<float> vm(pm.padded_size);
    memory_mapper<float> map(vm, pm, devices);

    cudaSetDevice(0);
    int block_size = 256;
    int chunk_size = round_up(N, num_gpus) / num_gpus;
    int grid_size = round_up(chunk_size, block_size) / block_size;

    std::cout << "Chunk Size = " << chunk_size << std::endl;
    std::cout << "Padded Count = " << pm.padded_size / sizeof(float) << std::endl;
    std::cout << "Striped Count = " << pm.stripe_size / sizeof(float) << std::endl;

    std::vector<std::thread> threads;
    my_timer_t t;

    nvtxRangePushA("vector_add");
    t.begin();
    for (int i = 0; i < num_gpus; ++i) {
        threads.push_back(std::thread([&, i]() {
        cudaSetDevice(i);

        auto input_begin = map.data() + (chunk_size * i);
        int elements_to_process = (i+1) * chunk_size;
        kernel<<<grid_size, block_size, 0, infos[i].stream>>>(input_begin, chunk_size * i, elements_to_process);
        cudaEventRecord(infos[i].event, infos[i].stream);
        }));
    }

    for (auto& thread : threads)
        thread.join();

    for (int i = 0; i < num_gpus; i++)
        cudaStreamWaitEvent(master_stream, infos[i].event, 0);

    cudaStreamSynchronize(master_stream);

    t.end();
    nvtxRangePop();

    std::cout << "Elapsed Time = " << t.milliseconds() << std::endl;

    return EXIT_SUCCESS;
}