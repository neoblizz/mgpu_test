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

// Use occupancy calculation for better occupancy.
int minimum_grid_size;
int max_active_blocks;
int block_size;

template <typename op_t>
__global__ void parallel_for(int* array, 
                             op_t apply, 
                             std::size_t length, 
                             std::size_t offset) {
  // const std::size_t STRIDE = (size_t)blockDim.x * gridDim.x;
  // std::size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
  // while (i < length) {
  //   apply(array[offset + i]);
  //   i += STRIDE;
  // }

  // Can be unrolled better, possibly nice for shared_memory optimization.
  #pragma unroll
  for(std::size_t i = blockDim.x * blockIdx.x + threadIdx.x; 
      i < length; 
      i=i+(blockDim.x * gridDim.x)) {
    apply(array[offset + i]);
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
    devices.push_back(i);
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&info.event, cudaEventDisableTiming);
    infos.push_back(info);
  }
  cudaSetDevice(0);
}

void occupancy() {

  auto dummy = [=] __device__ (int& vertex) {
    vertex = -1;
  };
  
  cudaOccupancyMaxPotentialBlockSize(&minimum_grid_size, &block_size, parallel_for<decltype(dummy)>, 0, 0);
  
  // Calculate theoretical occupancy.
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, parallel_for<decltype(dummy)>, block_size, 0);

  int curr_device;
  cudaDeviceProp props;
  cudaGetDevice(&curr_device);
  cudaGetDeviceProperties(&props, curr_device);

  float occupancy = (max_active_blocks * block_size / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  std::cout << "Block Size: " << block_size << std::endl;
  std::cout << "Minimum Grid Size: " << minimum_grid_size << std::endl;
  std::cout << "Maximum Active Blocks per SM: " << max_active_blocks << std::endl;
  std::cout << "Theoretical Occupancy: " << occupancy << std::endl;
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
  int n_nnz = g->n_nnz;
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
  physical_memory<int> pm_randoms(n_rows * num_gpus * sizeof(int), devices);
  // physical_memory<int> pm_randoms(n_rows * sizeof(int), devices);
  virtual_memory<int> vm_randoms(pm_randoms.padded_size);
  memory_mapper<int> map_randoms(vm_randoms, pm_randoms, devices);
  randoms = map_randoms.data();

  for(int i = 0; i < num_gpus; i++)
    cudaMemcpy(randoms + (n_rows * i), h_randoms, n_rows * sizeof(int), cudaMemcpyHostToDevice);

  // NO DUPLICATION::
  // cudaMemcpy(randoms, h_randoms, n_rows * sizeof(int), cudaMemcpyHostToDevice);

  // --
  // Run
  for(int dev = 0; dev < num_gpus; dev++) {
    cudaSetDevice(dev);  
    cudaDeviceSynchronize();
  }

  cudaSetDevice(0);

  my_timer_t t;
  std::vector<float> per_iteration_times;

  int* indptr = g->indptr;
  int* indices = g->indices;
  float* data = g->data;
  int iteration = 0;
  
  nvtxRangePushA("color:: while(iteration)");
  
  
  while(iteration < 29) {
    int chunk_size  = (n_rows + num_gpus - 1) / num_gpus;
    t.begin();
    
    #pragma omp parallel for num_threads(num_gpus)
    for(int dev = 0 ; dev < num_gpus ; dev++) {

        auto fn = [indptr, indices, data, colors, randoms, iteration, dev, n_rows, n_nnz] __device__(int& vertex) {
        if(vertex == -1) return;
        
        // weird device-aware access.
        // can be hidden behind graph API.
        int start  = indptr[vertex + ((n_rows + 1) * dev)];
        int end    = indptr[vertex + 1 + ((n_rows + 1) * dev)];
        int degree = end - start;

        bool colormax = true;
        bool colormin = true;
        int color     = iteration * 2;

        for (int i = 0; i < degree; i++) {
          // weird device-aware access.
          // can be hidden behind graph API.
          int u = indices[start + i + (n_nnz * dev)];

          if (colors[u] != -1 && (colors[u] != color + 1) && (colors[u] != color + 2) || (vertex == u))
            continue;
          if (randoms[vertex + (n_rows * dev)] <= randoms[u + (n_rows * dev)])  // weird device-aware access.
            colormax = false;
          if (randoms[vertex + (n_rows * dev)] >= randoms[u + (n_rows * dev)]) // weird device-aware access.
            colormin = false;
          
          if(!colormax && !colormin) return; // optimization
        }

        if (colormax) {
          colors[vertex] = color + 1;
          vertex = -1;
        } else if (colormin) {
          colors[vertex] = color + 2;
          vertex = -1;
        }
      };
      
      cudaSetDevice(dev);

      auto offset = chunk_size * dev;
      auto length = chunk_size;
      if(dev == num_gpus - 1) 
        length = n_rows - (chunk_size * (num_gpus - 1)) ;

      parallel_for<<<minimum_grid_size, 
                     block_size, 
                     0, infos[dev].stream>>>(
                       input, fn, length, offset);

      cudaEventRecord(infos[dev].event, infos[dev].stream);
    }
    
    for(int dev = 0; dev < num_gpus; dev++)
      cudaStreamWaitEvent(master_stream, infos[dev].event, 0);

    cudaStreamSynchronize(master_stream);
    t.end();

    iteration++;
    
    per_iteration_times.push_back(t.milliseconds());
    std::cout << t.milliseconds() << std::endl;
  }

  // Pop -> color:: while(iteration)
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

  // enable_peer_access();  // we don't need this.
  create_contexts();
  occupancy();
  read_binary(inpath, &h_graph);

  graph_t d_graph;
  int num_gpus = get_num_gpus();

  // IMPORTANT TODO!!!! We can do slightly better by 
  // declaring read only accesses for some stuff.
  // Like graphs, and randoms array.

  // Memory mapping for row pointers.
  physical_memory<int> pm_indptr((h_graph.n_rows + 1) * num_gpus * sizeof(int) , devices);
  // physical_memory<int> pm_indptr((h_graph.n_rows + 1) * sizeof(int) , devices);
  virtual_memory<int> vm_indptr(pm_indptr.padded_size);
  memory_mapper<int> map_indptr(vm_indptr, pm_indptr, devices);

  // Memory mapping for indices.
  physical_memory<int> pm_indices(h_graph.n_nnz * sizeof(int) * num_gpus , devices);  
  // physical_memory<int> pm_indices(h_graph.n_nnz * sizeof(int), devices);
  virtual_memory<int> vm_indices(pm_indices.padded_size);
  memory_mapper<int> map_indices(vm_indices, pm_indices, devices);

  // Memory mapping for data.
  physical_memory<float> pm_data(h_graph.n_nnz * sizeof(float) * num_gpus , devices);
  // physical_memory<float> pm_data(h_graph.n_nnz * sizeof(float), devices);
  virtual_memory<float> vm_data(pm_data.padded_size);
  memory_mapper<float> map_data(vm_data, pm_data, devices);

  d_graph.n_rows = h_graph.n_rows;
  d_graph.n_cols = h_graph.n_cols;
  d_graph.n_nnz = h_graph.n_nnz;
  d_graph.indptr = map_indptr.data();
  d_graph.indices = map_indices.data();
  d_graph.data = map_data.data();

  // Instead of one-copy of graph for the entire system, we create
  // graph that is num_gpus times bigger, and copy it to each device.
  for(int i = 0; i < num_gpus; i++) {
    cudaMemcpy(d_graph.indptr + ((d_graph.n_rows + 1) * i), h_graph.indptr, (h_graph.n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.indices + (d_graph.n_nnz * i), h_graph.indices, h_graph.n_nnz * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy(d_graph.data + (d_graph.n_nnz * i), h_graph.data, h_graph.n_nnz * sizeof(float), cudaMemcpyHostToDevice);
  }

  // NO DUPLICATION::
  // cudaMemcpy(d_graph.indptr, h_graph.indptr, (h_graph.n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_graph.indices, h_graph.indices, h_graph.n_nnz * sizeof(int), cudaMemcpyHostToDevice );
  // cudaMemcpy(d_graph.data, h_graph.data, h_graph.n_nnz * sizeof(float), cudaMemcpyHostToDevice);

  free(h_graph.indptr);
  free(h_graph.indices);
  free(h_graph.data);
  
  std::cout << "color | num_gpus: " << num_gpus << " vertices: " << d_graph.n_rows << std::endl;

  int num_iters = 4;
  for(int i = 0; i < num_iters; i++)
    do_test(&d_graph);

  d_graph.indptr = nullptr;
  d_graph.indices = nullptr;
  d_graph.data = nullptr;
  
  std::cout << "-----" << std::endl;
  return EXIT_SUCCESS;
}