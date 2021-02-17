#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

// #define LOCAL_COPY
// #define MANAGED_MEMORY

// template <typename T>
// class chunked_ptr {

//   int num_gpus = 1;
//   cudaGetDeviceCount(&num_gpus);
  
//   T* ptrs[num_gpus];
//   int n;
  
//   chunked_ptr(int n_, T* _ptr) : n(n_) {
//     auto chunk_size = (n + num_gpus - 1) / num_gpus;
    
//   }
// }


void do_test(int num_arguments, char** argument_array) {
  srand(112233);
  
  // --
  // Data
  
  int n_rows  = 4000000;
  int n_cols  = 4000000;
  int degree  = 50;
  int max_val = 100000;
  
  thrust::host_vector<int> h_indptr(n_rows + 1);
  thrust::host_vector<int> h_indices(n_rows * degree);
  thrust::host_vector<int> h_data(n_cols);
  
  h_indptr[0] = 0;
  for(int i = 1; i < n_rows + 1; i++)      h_indptr[i]  = h_indptr[i - 1] + degree;
  for(int i = 0; i < n_rows * degree; i++) h_indices[i] = rand() % n_cols;
  for(int i = 0; i < n_cols; i++)          h_data[i]    = rand() % max_val;

  thrust::device_vector<int> indptr  = h_indptr;
  thrust::device_vector<int> indices = h_indices;
  thrust::device_vector<int> data    = h_data;


#ifdef MANAGED_MEMORY

  int* g_indptr_ptr;
  cudaMallocManaged(&g_indptr_ptr, h_indptr.size() * sizeof(int));
  cudaMemcpy(g_indptr_ptr, indptr.data().get(), h_indptr.size() * sizeof(int), cudaMemcpyDeviceToDevice);
  
  int* g_indices_ptr;
  cudaMallocManaged(&g_indices_ptr, h_indices.size() * sizeof(int));
  cudaMemcpy(g_indices_ptr, indices.data().get(), h_indices.size() * sizeof(int), cudaMemcpyDeviceToDevice);
  
  int* g_data_ptr;
  cudaMallocManaged(&g_data_ptr, h_data.size() * sizeof(int));
  cudaMemcpy(g_data_ptr, data.data().get(), h_data.size() * sizeof(int), cudaMemcpyDeviceToDevice);

#else

  int* g_indptr_ptr  = indptr.data().get();
  int* g_indices_ptr = indices.data().get();
  int* g_data_ptr    = data.data().get();
  
#endif
  
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

#ifdef MANAGED_MEMORY
  for(int i = 0; i < num_gpus; i++) {
    cudaMemAdvise(g_indptr_ptr, h_indptr.size() * sizeof(int), cudaMemAdviseSetReadMostly, i);
    cudaMemAdvise(g_indices_ptr, h_indices.size() * sizeof(int), cudaMemAdviseSetReadMostly, i);
    cudaMemAdvise(g_data_ptr, h_data.size() * sizeof(int), cudaMemAdviseSetReadMostly, i);
  }
#endif
  
#ifdef LOCAL_COPY
  // Copy the datastructures to local GPUs
  int* indptr_ptrs[num_gpus];
  int* indices_ptrs[num_gpus];
  int* data_ptrs[num_gpus];
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    
    int* l_indptr_ptr;
    cudaMalloc((void**)&l_indptr_ptr, h_indptr.size() * sizeof(int));
    cudaMemcpy(l_indptr_ptr, g_indptr_ptr, h_indptr.size() * sizeof(int), cudaMemcpyDeviceToDevice);
    indptr_ptrs[i] = l_indptr_ptr;

    int* l_indices_ptr;
    cudaMalloc((void**)&l_indices_ptr, h_indices.size() * sizeof(int));
    cudaMemcpy(l_indices_ptr, g_indices_ptr, h_indices.size() * sizeof(int), cudaMemcpyDeviceToDevice);
    indices_ptrs[i] = l_indices_ptr;

    int* l_data_ptr;
    cudaMalloc((void**)&l_data_ptr, h_data.size() * sizeof(int));
    cudaMemcpy(l_data_ptr, g_data_ptr, h_data.size() * sizeof(int), cudaMemcpyDeviceToDevice);
    data_ptrs[i] = l_data_ptr;
    
    cudaDeviceSynchronize();
  }
  cudaSetDevice(0);
  cudaDeviceSynchronize();
#endif
  
  // --
  // Setup devices
  
  cudaSetDevice(0);
  cudaStream_t master_stream;
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);

  auto chunk_size = (n_rows + num_gpus - 1) / num_gpus;
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
  // Run filter
  
  for(int iter = 0 ; iter < 10 ; iter++) {

    // --
    // Initialize Frontier
    
    thrust::host_vector<int> h_input(n_rows);
    thrust::host_vector<int> h_output(n_rows);
    for(int i = 0; i < n_rows; i++) h_input[i] = i;
    for(int i = 0; i < n_rows; i++) h_output[i] = -1;

    thrust::device_vector<int> input   = h_input;
    thrust::device_vector<int> output  = h_output;
    
    // --
    // Run
    
    cudaSetDevice(0);  
    cudaDeviceSynchronize();
    
    int new_sizes[num_gpus];
    
    nvtxRangePushA("thrust_work");
    
    #pragma omp parallel for num_threads(num_gpus)
    for(int i = 0 ; i < num_gpus ; i++) {
      cudaSetDevice(i);

  #ifdef LOCAL_COPY
      int* indptr_ptr  = indptr_ptrs[i];
      int* indices_ptr = indices_ptrs[i];
      int* data_ptr    = data_ptrs[i];
  #else
      int* indptr_ptr  = g_indptr_ptr;
      int* indices_ptr = g_indices_ptr;
      int* data_ptr    = g_data_ptr;
  #endif

      auto fn = [indptr_ptr, indices_ptr, data_ptr] __host__ __device__(int const& i) -> bool {      
        int start  = indptr_ptr[i];
        int end    = indptr_ptr[i + 1];
        int degree = end - start;
        
        int acc = 0;
        for(int i = 0; i < degree; i++) {
          int idx = indices_ptr[start + i];
          acc += data_ptr[idx];
        }
        return acc % 2 == 0;
      };
      
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

    thrust::host_vector<int> ttmp = input;
    thrust::copy(ttmp.begin(), ttmp.begin() + 32, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  do_test(argc, argv);
  return EXIT_SUCCESS;
}
