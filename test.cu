#include <cstdlib>  // EXIT_SUCCESS
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
  
  int n = 200000;
  
  thrust::host_vector<int> h_input(n);
  thrust::host_vector<int> h_output(n);
  
  for(int i = 0; i < n; i++) h_input[i] = i;
  thrust::fill(thrust::host, h_output.begin(), h_output.end(), -1);
    
  // --
  // Setup data
  
  int num_gpus = 4;
  
  cudaSetDevice(0);
  thrust::device_vector<int> input0  = h_input;
  thrust::device_vector<int> toutput0 = h_output;
  thrust::device_vector<int> koutput0 = h_output;

  cudaSetDevice(1);
  thrust::device_vector<int> input1  = h_input;
  thrust::device_vector<int> toutput1 = h_output;
  thrust::device_vector<int> koutput1 = h_output;

  cudaSetDevice(2);
  thrust::device_vector<int> input2  = h_input;
  thrust::device_vector<int> toutput2 = h_output;
  thrust::device_vector<int> koutput2 = h_output;

  cudaSetDevice(3);
  thrust::device_vector<int> input3  = h_input;
  thrust::device_vector<int> toutput3 = h_output;
  thrust::device_vector<int> koutput3 = h_output;

  std::vector<thrust::device_vector<int>*> all_inputs;
  all_inputs.push_back(&input0);
  all_inputs.push_back(&input1);
  all_inputs.push_back(&input2);
  all_inputs.push_back(&input3);
  
  std::vector<thrust::device_vector<int>*> all_outputs_thrust;
  all_outputs_thrust.push_back(&toutput0);
  all_outputs_thrust.push_back(&toutput1);
  all_outputs_thrust.push_back(&toutput2);
  all_outputs_thrust.push_back(&toutput3);

  std::vector<thrust::device_vector<int>*> all_outputs_kernel;
  all_outputs_kernel.push_back(&koutput0);
  all_outputs_kernel.push_back(&koutput1);
  all_outputs_kernel.push_back(&koutput2);
  all_outputs_kernel.push_back(&koutput3);

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
  // Run
  
  auto fn = [=] __host__ __device__(int const& i) -> bool {
    int acc = 0;
    for(int ii = 0; ii < i; ii++)
      acc += ii;
    
    return acc % 2 == 0 ? 0 : 1;
  };
  
  // Thrust
  nvtxRangePushA("thrust_work");
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);

    thrust::transform(
      thrust::cuda::par.on(infos[i].stream),
      all_inputs[i]->begin(),
      all_inputs[i]->end(),
      all_outputs_thrust[i]->begin(),
      fn
    );
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  
  nvtxRangePop();
  
  // Kernel
  nvtxRangePushA("kernel_work");
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);
    fn_kernel<<<(n + 255) / 256, 256, 0, infos[i].stream>>>(
      n, 
      all_inputs[i]->data().get(),
      all_outputs_kernel[i]->data().get()
    );

    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  
  nvtxRangePop();
  
  cudaSetDevice(0);
  thrust::host_vector<int> ttmp = *all_outputs_thrust[0];
  thrust::copy(ttmp.begin(), ttmp.begin() + 100, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  
  thrust::host_vector<int> ktmp = *all_outputs_kernel[0];
  thrust::copy(ktmp.begin(), ktmp.begin() + 100, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  do_test(argc, argv);
  return EXIT_SUCCESS;
}
