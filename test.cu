#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include <chrono>
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

// #define MANAGED

using namespace std::chrono;

int n_rows;
int n_cols;
int n_nnz;

int* h_indptr;
int* h_indices;
float* h_data;

int* g_indptr;
int* g_indices;
float* g_data;

struct gpu_info {
  cudaStream_t stream;
  cudaEvent_t  event;
};

std::vector<gpu_info> infos;

cudaStream_t master_stream;

int get_num_gpus() {
  int num_gpus = -1;
  cudaGetDeviceCount(&num_gpus);
  return num_gpus;
}

void enable_peer_access() {
  int num_gpus = get_num_gpus();
  
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    for(int j = 0; j < num_gpus; j++) {
      if(i == j) 
        continue;
      cudaDeviceEnablePeerAccess(j, 0);
    }
  }
  
  cudaSetDevice(0);
}

void create_contexts() {
  int num_gpus = get_num_gpus();
  
  cudaSetDevice(0);
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  
  for(int i = 0 ; i < num_gpus ; i++) {
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreate(&info.event);
    infos.push_back(info);
  }
  
  cudaSetDevice(0);
}

void read_binary(std::string filename) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  auto err = fread(&n_rows, sizeof(int), 1, file);
  err = fread(&n_cols, sizeof(int), 1, file);
  err = fread(&n_nnz,  sizeof(int), 1, file);

  h_indptr  = (int*  )malloc((n_rows + 1) * sizeof(int));
  h_indices = (int*  )malloc(n_nnz        * sizeof(int));
  h_data    = (float*)malloc(n_nnz        * sizeof(float));

  err = fread(h_indptr,  sizeof(int),   n_rows + 1, file);
  err = fread(h_indices, sizeof(int),   n_nnz,      file);
  err = fread(h_data,    sizeof(float), n_nnz,      file);

#ifdef MANAGED
  cudaMallocManaged(&g_indptr,  (n_rows + 1) * sizeof(int));
  cudaMallocManaged(&g_indices, n_nnz        * sizeof(int));
  cudaMallocManaged(&g_data,    n_nnz        * sizeof(float));
#else
  cudaMalloc(&g_indptr, (n_rows + 1) * sizeof(int));
  cudaMalloc(&g_indices, n_nnz       * sizeof(int));
  cudaMalloc(&g_data,    n_nnz       * sizeof(float));
#endif

  cudaMemcpy(g_indptr, h_indptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g_indices, h_indices, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g_data, h_data, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

#ifdef MANAGED
  for(int i = 0; i < get_num_gpus(); i++) {
    cudaMemAdvise(g_indptr, (n_rows + 1) * sizeof(int), cudaMemAdviseSetReadMostly, i);
    cudaMemAdvise(g_indices, n_nnz * sizeof(int), cudaMemAdviseSetReadMostly, i);
    cudaMemAdvise(g_data, n_nnz * sizeof(float), cudaMemAdviseSetReadMostly, i);
  }
#endif  
}

void do_test() {
  
  int num_gpus    = get_num_gpus();
  int chunk_size  = (n_rows + num_gpus - 1) / num_gpus;

  // --
  // initialize frontier
  
  thrust::host_vector<int> h_input(n_rows);
  thrust::host_vector<int> h_output(n_rows);
  for(int i = 0; i < n_rows; i++) h_input[i] = i;
  for(int i = 0; i < n_rows; i++) h_output[i] = -1;

  thrust::device_vector<int> input   = h_input;
  thrust::device_vector<int> output  = h_output;
  
  // --
  // initialize data structures
  
//   int* h_color = (int*)malloc(n_rows * sizeof(int));
  
//   int* g_color;
// #ifdef MANAGED
//   cudaMallocManaged(&g_color, n_rows * sizeof(int));
// #else
//   cudaMalloc(&g_color, n_rows * sizeof(int));
// #endif
//   cudaMemcpy(g_color, h_color, n_rows * sizeof(int), cudaMemcpyHostToDevice);
  
  // --
  // Run
  
  cudaSetDevice(0);  
  cudaDeviceSynchronize();
  
  int new_sizes[num_gpus];
  
  int* indptr  = g_indptr;
  int* indices = g_indices;
  float* data  = g_data;
  // int* color   = g_color;
  
  nvtxRangePushA("thrust_work");
  
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0 ; i < num_gpus ; i++) {
    cudaSetDevice(i);

    auto fn = [indptr, indices, data] __host__ __device__(int const& i) -> bool {      
      int start  = indptr[i];
      int end    = indptr[i + 1];
      int degree = end - start;
      
      int acc = 0;
      for(int i = 0; i < degree; i++) {
        int idx = indices[start + i];
        acc += (int)data[idx];
      }
      bool val = acc % 2 == 0;
      // color[i] = (int)val;
      return val;
    };
    
    auto input_begin  = input.begin() + chunk_size * i;
    auto input_end    = input.begin() + chunk_size * (i + 1);
    auto output_begin = output.begin() + chunk_size * i;
    if(i == num_gpus - 1) input_end = input.end();
    
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
  
  input.resize(total_length);
  nvtxRangePop();
  
  // Log
  thrust::host_vector<int> r_input = input;
  thrust::copy(r_input.begin(), r_input.begin() + 32, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  
  cudaSetDevice(0);
}

int main(int argc, char** argv) {
  std::string inpath = argv[1];
  
  enable_peer_access();
  create_contexts();
  read_binary(inpath);

  int num_gpus = get_num_gpus();

  auto t1 = high_resolution_clock::now();
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i); cudaDeviceSynchronize();
  }
  
  int num_iters = 4;
  for(int i = 0; i < num_iters; i++)
    do_test();
  
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i); cudaDeviceSynchronize();
  }
  
  auto elapsed = high_resolution_clock::now() - t1;
  long long ms = duration_cast<microseconds>(elapsed).count();
  std::cout << "elapsed: " << ms << std::endl;
  
  return EXIT_SUCCESS;
}
