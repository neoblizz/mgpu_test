#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

// #define do_copy_if

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
  srand(112233);
  
  // --
  // Create data
  
  int n = 1000000;
  
  thrust::host_vector<int> h_input(n);
  thrust::host_vector<int> h_output(n);
  
  for(int i = 0; i < n; i++) h_input[i] = rand() % 100000;
  thrust::fill(thrust::host, h_output.begin(), h_output.end(), -1);

  thrust::device_vector<int> input  = h_input;
  thrust::device_vector<int> output = h_output;
  // thrust::device_vector<int> output_kernel = h_output;

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
  
  // int num_gpus = 4;
  
  // cudaSetDevice(0);
  // thrust::device_vector<int> input0  = h_input;
  // thrust::device_vector<int> output_thrust0 = h_output;
  // thrust::device_vector<int> output_kernel0 = h_output;

  // cudaSetDevice(1);
  // thrust::device_vector<int> input1  = h_input;
  // thrust::device_vector<int> output_thrust1 = h_output;
  // thrust::device_vector<int> output_kernel1 = h_output;

  // cudaSetDevice(2);
  // thrust::device_vector<int> input2  = h_input;
  // thrust::device_vector<int> output_thrust2 = h_output;
  // thrust::device_vector<int> output_kernel2 = h_output;

  // cudaSetDevice(3);
  // thrust::device_vector<int> input3  = h_input;
  // thrust::device_vector<int> output_thrust3 = h_output;
  // thrust::device_vector<int> output_kernel3 = h_output;

  // std::vector<thrust::device_vector<int>*> all_inputs;
  // all_inputs.push_back(&input0);
  // all_inputs.push_back(&input1);
  // all_inputs.push_back(&input2);
  // all_inputs.push_back(&input3);
  
  // std::vector<thrust::device_vector<int>*> all_outputs_thrust;
  // all_outputs_thrust.push_back(&output_thrust0);
  // all_outputs_thrust.push_back(&output_thrust1);
  // all_outputs_thrust.push_back(&output_thrust2);
  // all_outputs_thrust.push_back(&output_thrust3);

  // std::vector<thrust::device_vector<int>*> all_outputs_kernel;
  // all_outputs_kernel.push_back(&output_kernel0);
  // all_outputs_kernel.push_back(&output_kernel1);
  // all_outputs_kernel.push_back(&output_kernel2);
  // all_outputs_kernel.push_back(&output_kernel3);

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

  cudaSetDevice(0);  
  
  
#ifdef do_copy_if
  auto fn = [=] __host__ __device__(int const& i) -> bool {
    int acc = 0;
    for(int ii = 0; ii < i; ii++)
      acc += ii;
    
    return acc % 2 != 0;
  };

  nvtxRangePushA("thrust_work");
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);

    auto input_begin  = input.begin() + chunk_size * i;
    auto input_end    = input.begin() + chunk_size * (i + 1);
    // auto output_begin = output.begin() + chunk_size * i;
    thrust::device_vector<int> local_output(chunk_size);
    
    if(i == num_gpus - 1) input_end = input.end();
    
    thrust::copy_if(
      thrust::cuda::par.on(infos[i].stream),
      input_begin,
      input_end,
      local_output.begin(),
      fn
    );
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  for(int i = 0; i < num_gpus; i++) {cudaSetDevice(i); cudaDeviceSynchronize();}
  nvtxRangePop();

  thrust::host_vector<int> ttmp = output;
  thrust::copy(ttmp.begin(), ttmp.begin() + 100, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
#else
  
  auto fn = [=] __host__ __device__(int const& i) -> int {
    int acc = 0;
    for(int ii = 0; ii < i; ii++)
      acc += ii;
    
    return acc % 2 == 0 ? -1 : i;
  };
  
  auto fn2 = [=] __host__ __device__(int const& i) -> int {
    return i >= 0;
  };
  
  nvtxRangePushA("thrust_work");
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);

    auto input_begin  = input.begin() + chunk_size * i;
    auto input_end    = input.begin() + chunk_size * (i + 1);
    auto output_begin = output.begin() + chunk_size * i;
    
    if(i == num_gpus - 1) input_end = input.end();
    
    thrust::transform(
      thrust::cuda::par.on(infos[i].stream),
      input_begin,
      input_end,
      output_begin,
      fn
    );
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  
  cudaSetDevice(0);
  auto new_end = thrust::copy_if(
    thrust::cuda::par.on(infos[0].stream),
    output.begin(),
    output.end(),
    input.begin(),
    fn2
  );
  auto new_size = thrust::distance(input.begin(), new_end);
  input.resize(new_size);
  
  cudaEventRecord(infos[0].event, infos[0].stream);
  cudaStreamWaitEvent(master_stream, infos[0].event, 0);
  
  for(int i = 0; i < num_gpus; i++) {cudaSetDevice(i); cudaDeviceSynchronize();}
  nvtxRangePop();
  thrust::host_vector<int> ttmp = input;
  thrust::copy(ttmp.begin(), ttmp.begin() + 100, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
#endif

}

int main(int argc, char** argv) {
  for(int i = 0 ; i < 10 ; i++)
    do_test(argc, argv);
  return EXIT_SUCCESS;
}
