#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

__global__ void fn_kernel(int n, int* x, int* y) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int i   = x[idx];
    int acc = 0;
    for(int ii = 0; ii < i; ii++) {
      acc += ii;
    }
    y[i] = (int)(acc % 2);
  }
}

void do_test(int num_arguments, char** argument_array) {
  
  // --
  // Create data
  
  int n = 400000;
  
  thrust::host_vector<int> h_input(n);
  thrust::host_vector<int> h_output(n);
  
  for(int i = 0; i < n; i++) h_input[i] = rand() % 1000000;
  thrust::fill(thrust::host, h_output.begin(), h_output.end(), -1);

  thrust::device_vector<int> input = h_input;
  thrust::device_vector<int> output_thrust = h_output;
  thrust::device_vector<int> output_kernel = h_output;

  // --
  // Setup data
  
  int num_gpus = 1;
  cudaGetDeviceCount(&num_gpus);
  
  // Peer access
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    for(int j = 0; j < num_gpus; j++) {
      if(i == j) continue;
      cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  // --
  // Setup devices
  
  cudaSetDevice(0);
  cudaStream_t master_stream;
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);

  auto chunk_size = n / num_gpus;
  std::cout << "num_gpus  : " << num_gpus << std::endl;
  std::cout << "chunk_size: " << chunk_size << std::endl;
  
  struct gpu_info {
    cudaStream_t stream;
    cudaEvent_t  event;
  };
  
  std::vector<gpu_info> infos;
  
  for(int i = 0 ; i < num_gpus ; i++) {
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreate(&info.event);
    infos.push_back(info);
  }
  
  // --
  // Thrust
  
  cudaSetDevice(0);
  
  auto fn = [=] __host__ __device__(int const& i) -> bool {
    int acc = 0;
    for(int ii = 0; ii < i; ii++)
      acc += ii;
    
    return (i + acc) % 2 == 0;
  };
  
  nvtxRangePushA("thrust_work");
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);

    auto input_begin  = input.begin() + chunk_size * i;
    auto input_end    = input.begin() + chunk_size * (i + 1);
    auto output_begin = output_thrust.begin() + chunk_size * i;
    
    if(i == num_gpus - 1) input_end = input.end();
    
    thrust::copy_if(
      thrust::cuda::par.on(infos[i].stream),
      input_begin,
      input_end,
      output_begin,
      fn
    );
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  for(int i = 0; i < num_gpus; i++) {cudaSetDevice(i); cudaDeviceSynchronize();}
  nvtxRangePop();
  
  // // --
  // // Kernel
  
  // cudaSetDevice(0);
  
  // nvtxRangePushA("kernel_work");
  // for(int i = 0 ; i < num_gpus ; i++) {
  //   cudaSetDevice(i);
  //   fn_kernel<<<(n + 255) / 256, 256, 0, infos[i].stream>>>(
  //     n, 
  //     input.data().get(),
  //     output_kernel.data().get()
  //   );

  //   cudaEventRecord(infos[i].event, infos[i].stream);
  // }
  
  // for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  // for(int i = 0; i < num_gpus; i++) {cudaSetDevice(i); cudaDeviceSynchronize();}
  // nvtxRangePop();
  
  thrust::host_vector<int> ttmp = output_thrust;
  thrust::copy(ttmp.begin(), ttmp.begin() + 40, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  
  // thrust::host_vector<int> ktmp = output_kernel;
  // thrust::copy(ktmp.begin(), ktmp.begin() + 40, std::ostream_iterator<int>(std::cout, " "));
  // std::cout << std::endl;
}

int main(int argc, char** argv) {
  for(int i = 0 ; i < 10 ; i++)
    do_test(argc, argv);
  return EXIT_SUCCESS;
}
