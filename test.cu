#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"


void do_test(int num_arguments, char** argument_array) {
  srand(112233);
  
  // --
  // Create data
  
  int n = 16;
  
  thrust::host_vector<int> h_input(n);
  thrust::host_vector<int> h_output(n);
  
  for(int i = 0; i < n; i++) 
    h_input[i] = rand() % 1000000;
  
  thrust::fill(thrust::host, h_output.begin(), h_output.end(), -1);

  thrust::device_vector<int> input  = h_input;
  thrust::device_vector<int> output = h_output;

  // --
  // Setup data
  
  int num_gpus = 1;
  cudaGetDeviceCount(&num_gpus);
  
  // Peer access
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    for(int j = 0; j < num_gpus; j++) {
      if(i == j) 
        continue;
      cudaDeviceEnablePeerAccess(j, 0);
    }
  }
  
  // --
  // Setup devices
  
  cudaSetDevice(0);
  cudaStream_t master_stream;
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);

  auto chunk_size = (n + num_gpus - 1) / num_gpus;
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
  
  cudaDeviceSynchronize();
  
  int new_sizes[num_gpus];
  
  nvtxRangePushA("thrust_work");
  
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);

    auto input_begin  = input.begin() + chunk_size * i;
    auto input_end    = input.begin() + chunk_size * (i + 1);
    auto output_begin = output.begin() + chunk_size * i;
    
    if(i == num_gpus - 1)
      input_end = input.end();
    
    auto new_output_end = thrust::copy_if(
      thrust::cuda::par.on(infos[i].stream),
      input_begin,
      input_end,
      output_begin,
      fn
    );
    new_sizes[i] = (int)thrust::distance(output_begin, new_output_end);
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++)
    cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  cudaStreamSynchronize(master_stream);
  
  int total_length = 0;
  int offsets[num_gpus];
  offsets[0] = 0;
  for(int i = 1 ; i < num_gpus ; i++) offsets[i] = new_sizes[i - 1] + offsets[i - 1];
  for(int i = 0 ; i < num_gpus ; i++) total_length += new_sizes[i];

  // Reduce
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);

    auto output_begin = output.begin() + chunk_size * i;
    thrust::copy_n(
      thrust::cuda::par.on(infos[i].stream),
      output_begin, 
      new_sizes[i], 
      input.begin() + offsets[i]
    );
    
    cudaEventRecord(infos[i].event, infos[i].stream);
  }
  
  for(int i = 0; i < num_gpus; i++)
    cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  
  cudaStreamSynchronize(master_stream);

  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i); cudaDeviceSynchronize();
  }
  
  input.resize(total_length);
  
  nvtxRangePop();

  thrust::host_vector<int> tmp = output;
  thrust::copy(tmp.begin(), tmp.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  thrust::host_vector<int> ttmp = input;
  thrust::copy(ttmp.begin(), ttmp.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  for(int i = 0 ; i < 10 ; i++)
    do_test(argc, argv);
  return EXIT_SUCCESS;
}
