#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include <cstdlib>  // EXIT_SUCCESS
#include <thread>
#include <iostream>
#include "omp.h"

#include <nvToolsExt.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include "memory.hxx"

#define FORALL_BLOCKSIZE 256
#define FORALL_GRIDSIZE 256

template <typename array_t, typename size_t, typename op_t>
__global__ void parallel_for(array_t array, 
                             op_t apply, 
                             size_t length, 
                             size_t offset) {
  const size_t STRIDE = (size_t)blockDim.x * gridDim.x;
  size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    // printf("id = %u\n", i + offset);
    apply(array[offset + i]);
    i += STRIDE;
  }
}

struct graph_t {
  int n_rows;
  int n_cols;
  int n_nnz;

  int* indptr;
  int* indices;
  float* data;
};

struct gpu_info {
  cudaStream_t stream;
  cudaEvent_t  event;
};

std::vector<gpu_info> infos;
std::vector<int> devices;
cudaStream_t master_stream;

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

int get_num_gpus() {
  int num_gpus = -1;
  cudaGetDeviceCount(&num_gpus);
  return num_gpus;
}

void create_contexts() {
  int num_gpus = get_num_gpus();
  
  cudaSetDevice(0);
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  
  for(int i = 0 ; i < num_gpus ; i++) {
    devices.push_back(i);
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&info.event, cudaEventDisableTiming);
    infos.push_back(info);
  }
  
  cudaSetDevice(0);
}

void read_binary(std::string filename, graph_t* g) {
  FILE* file = fopen(filename.c_str(), "rb");
  
  auto err = fread(&(g->n_rows), sizeof(int), 1, file);
  err = fread(&(g->n_cols), sizeof(int), 1, file);
  err = fread(&(g->n_nnz),  sizeof(int), 1, file);

  g->indptr  = (int*  )malloc((g->n_rows + 1) * sizeof(int));
  g->indices = (int*  )malloc(g->n_nnz        * sizeof(int));
  g->data    = (float*)malloc(g->n_nnz        * sizeof(float));

  err = fread(g->indptr,  sizeof(int),   g->n_rows + 1, file);
  err = fread(g->indices, sizeof(int),   g->n_nnz,      file);
  err = fread(g->data,    sizeof(float), g->n_nnz,      file);
}

void do_test(graph_t* g) {
  srand(123123123);
  
  int num_gpus = get_num_gpus();
  int n_rows = g->n_rows;
  // int n_nnz = g->n_nnz;
  // int n_cols = g->n_cols;

  // --
  // initialize frontier
  
  thrust::host_vector<int> h_input(n_rows);
  for(int i = 0; i < n_rows; i++) h_input[i] = i;

  int* input;

  // Memory mapping for frontier.
  physical_memory<int> pm_input(n_rows * sizeof(int), devices);
  virtual_memory<int> vm_input(pm_input.padded_size);
  memory_mapper<int> map_input(vm_input, pm_input, devices);
  input = map_input.data();

  cudaMemcpy(input, h_input.data(), n_rows * sizeof(int), cudaMemcpyHostToDevice);
  
  // --
  // initialize data structures
  int* colors;

  // Memory mapping for frontier.
  physical_memory<int> pm_colors(n_rows * sizeof(int) , devices);
  virtual_memory<int> vm_colors(pm_colors.padded_size);
  memory_mapper<int> map_colors(vm_colors, pm_colors, devices);
  colors = map_colors.data();

  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);  
    cudaDeviceSynchronize();
  }

  cudaMemset((void*)colors, -1, n_rows * sizeof(int));

  int* h_randoms = (int*)malloc(n_rows * sizeof(int));
  for(int i = 0; i < n_rows; i++) h_randoms[i] = rand() % n_rows;
  
  int* randoms;
  physical_memory<int> pm_randoms(n_rows * sizeof(int) , devices);
  virtual_memory<int> vm_randoms(pm_randoms.padded_size);
  memory_mapper<int> map_randoms(vm_randoms, pm_randoms, devices);
  randoms = map_randoms.data();
  cudaMemcpy(randoms, h_randoms, n_rows * sizeof(int), cudaMemcpyHostToDevice);

  // --
  // Run
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);  
    cudaDeviceSynchronize();
  }

  cudaSetDevice(0);

  my_timer_t t;
  std::vector<float> per_iteration_times;
  
  nvtxRangePushA("thrust_work");

  int* indptr = g->indptr;
  int* indices = g->indices;
  float* data = g->data;
  
  int iteration = 0;
  while(iteration < 29) {
    t.begin();

    auto fn = [indptr, indices, data, colors, randoms, iteration] __device__(int const& vertex) {
      if(vertex == -1) return -1;
      
      int start  = indptr[vertex];
      int end    = indptr[vertex + 1];
      int degree = end - start;

      bool colormax = true;
      bool colormin = true;
      int color     = iteration * 2;

      for (int i = 0; i < degree; i++) {
        int u = indices[start + i];

        if (colors[u] != -1 && (colors[u] != color + 1) && (colors[u] != color + 2) || (vertex == u))
          continue;
        if (randoms[vertex] <= randoms[u])
          colormax = false;
        if (randoms[vertex] >= randoms[u])
          colormin = false;
      }

      if (colormax) {
        colors[vertex] = color + 1;
        return -1;
      } else if (colormin) {
        colors[vertex] = color + 2;
        return -1;
      } else {
        return vertex;
      }
    };

    int chunk_size  = (n_rows + num_gpus - 1) / num_gpus;
    
    #pragma omp parallel for num_threads(num_gpus)
    for(int i = 0 ; i < num_gpus ; i++) {
      
      cudaSetDevice(i);
      
      // auto input_begin  = input + chunk_size * i;
      // auto input_end    = input + chunk_size * (i + 1);
      // if(i == num_gpus - 1) input_end = input + n_rows;

      auto offset = chunk_size * i;
      auto length = chunk_size;
      if(i == num_gpus - 1) length = n_rows - (chunk_size * (num_gpus - 1)) ;

      parallel_for<<<FORALL_GRIDSIZE, 
                     FORALL_BLOCKSIZE, 
                     0, infos[i].stream>>>(
                       input, fn, length, offset);

      cudaEventRecord(infos[i].event, infos[i].stream);
    }
    
    for(int i = 0; i < num_gpus; i++)
      cudaStreamWaitEvent(master_stream, infos[i].event, 0);

    cudaStreamSynchronize(master_stream);
      
    iteration++;
    t.end();
    per_iteration_times.push_back(t.milliseconds());
    std::cout << t.milliseconds() << std::endl;
  }
  nvtxRangePop();
  
  // Log
  thrust::host_vector<int> out(n_rows);
  cudaMemcpy(out.data(), input, n_rows * sizeof(int), cudaMemcpyDeviceToHost);

  thrust::copy(out.begin(), out.begin() + 32, std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  
  cudaSetDevice(0);

  float total_elapsed = 0;
  for (auto& n : per_iteration_times)
    total_elapsed += n;

  std::cout << "total_elapsed: " << total_elapsed << std::endl;
}

int main(int argc, char** argv) {
  std::string inpath = argv[1];
  graph_t h_graph;

  create_contexts();
  read_binary(inpath, &h_graph);

  graph_t d_graph;

  // Memory mapping for row pointers.
  physical_memory<int> pm_indptr((h_graph.n_rows + 1) * sizeof(int) , devices);
  virtual_memory<int> vm_indptr(pm_indptr.padded_size);
  memory_mapper<int> map_indptr(vm_indptr, pm_indptr, devices);

  // Memory mapping for indices.
  physical_memory<int> pm_indices(h_graph.n_nnz * sizeof(int) , devices);
  virtual_memory<int> vm_indices(pm_indices.padded_size);
  memory_mapper<int> map_indices(vm_indices, pm_indices, devices);

  // Memory mapping for data.
  physical_memory<float> pm_data(h_graph.n_nnz * sizeof(float) , devices);
  virtual_memory<float> vm_data(pm_data.padded_size);
  memory_mapper<float> map_data(vm_data, pm_data, devices);

  d_graph.n_rows = h_graph.n_rows;
  d_graph.n_cols = h_graph.n_cols;
  d_graph.n_nnz = h_graph.n_nnz;
  d_graph.indptr = map_indptr.data();
  d_graph.indices = map_indices.data();
  d_graph.data = map_data.data();

  cuMemcpyHtoD((CUdeviceptr)d_graph.indptr, (void*)h_graph.indptr, (h_graph.n_rows + 1) * sizeof(int));
  cuMemcpyHtoD((CUdeviceptr)d_graph.indices, (void*)h_graph.indices, h_graph.n_nnz * sizeof(int));
  cuMemcpyHtoD((CUdeviceptr)d_graph.data, (void*)h_graph.data, h_graph.n_nnz * sizeof(float));

  free(h_graph.indptr);
  free(h_graph.indices);
  free(h_graph.data);
  
  int num_gpus = get_num_gpus();
  std::cout << "color | num_gpus: " << num_gpus << " vertices: " << d_graph.n_rows << std::endl;

  int num_iters = 1;
  for(int i = 0; i < num_iters; i++)
    do_test(&d_graph);

  d_graph.indptr = nullptr;
  d_graph.indices = nullptr;
  d_graph.data = nullptr;
  
  std::cout << "-----" << std::endl;
  return EXIT_SUCCESS;
}