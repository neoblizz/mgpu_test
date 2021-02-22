#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include <cstdlib>  // EXIT_SUCCESS
#include "omp.h"
#include "nvToolsExt.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include <thrust/iterator/counting_iterator.h>
#include "thrust/random.h"

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

template <typename index_t, typename iterator_t>
void uniform_distribution(index_t begin, index_t end, iterator_t input) {
  using type_t = typename std::iterator_traits<iterator_t>::value_type;

  auto generate_random = [] __device__(int i) -> type_t {
    thrust::default_random_engine rng;
    rng.discard(i);
    return rng();
  };
  
  thrust::transform(thrust::make_counting_iterator(begin), thrust::make_counting_iterator(end), input, generate_random);
}

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
    cudaEventCreateWithFlags(&info.event, cudaEventDisableTiming);
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

  cudaMallocManaged(&g_indptr,  (n_rows + 1) * sizeof(int));
  cudaMallocManaged(&g_indices, n_nnz        * sizeof(int));
  cudaMallocManaged(&g_data,    n_nnz        * sizeof(float));

  cudaMemcpy(g_indptr, h_indptr, (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g_indices, h_indices, n_nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(g_data, h_data, n_nnz * sizeof(int), cudaMemcpyHostToDevice);

  // cudaMemAdviseSetReadMostly: The device argument is ignored for this advice.
  cudaMemAdvise(g_indptr, (n_rows + 1) * sizeof(int), cudaMemAdviseSetReadMostly, 0);
  cudaMemAdvise(g_indices, n_nnz * sizeof(int), cudaMemAdviseSetReadMostly, 0);
  cudaMemAdvise(g_data, n_nnz * sizeof(float), cudaMemAdviseSetReadMostly, 0);

  int num_gpus = get_num_gpus();

  // Prefetch the graph data to all devices:
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0; i < num_gpus; i++) {
    cudaMemAdvise(g_indptr, (n_rows + 1) * sizeof(int), cudaMemAdviseSetAccessedBy, i);
    cudaMemAdvise(g_indices, n_nnz * sizeof(int), cudaMemAdviseSetAccessedBy, i);
    cudaMemAdvise(g_data, n_nnz * sizeof(float), cudaMemAdviseSetAccessedBy, i);
    
    cudaMemPrefetchAsync(g_indptr, (n_rows + 1) * sizeof(int), i, 0);
    cudaMemPrefetchAsync(g_indices, n_nnz * sizeof(int), i, 0);
    cudaMemPrefetchAsync(g_data, n_nnz * sizeof(float), i, 0);
  }
}

void do_test() {
  srand(123123123);
  
  int num_gpus = get_num_gpus();

  // --
  // initialize frontier
  
  thrust::host_vector<int> h_input(n_rows);
  // thrust::host_vector<int> h_output(n_rows);
  for(int i = 0; i < n_rows; i++) h_input[i] = i;
  // for(int i = 0; i < n_rows; i++) h_output[i] = -1;

  // thrust::device_vector<int> input   = h_input;
  // thrust::device_vector<int> output  = h_output;
  int* input;
  cudaMallocManaged(&input, n_rows * sizeof(int));
  cudaMemcpy(input, h_input.data(), n_rows * sizeof(int), cudaMemcpyHostToDevice);
  
  
  // --
  // initialize data structures
  int* colors;
  cudaMallocManaged(&colors, n_rows * sizeof(int));
  thrust::fill(thrust::device, colors, colors + n_rows, -1);

  int* h_randoms = (int*)malloc(n_rows * sizeof(int));
  for(int i = 0; i < n_rows; i++) h_randoms[i] = rand() % n_rows;
  
  int* randoms;
  cudaMallocManaged(&randoms, n_rows * sizeof(int));
  cudaMemcpy(randoms, h_randoms, n_rows * sizeof(int), cudaMemcpyHostToDevice);

  // === cudaMemAdviseSetReadMostly: The device argument is ignored for this advice.
  cudaMemAdvise(randoms, n_rows * sizeof(int), cudaMemAdviseSetReadMostly, 0);

  int partitioned = (n_rows + num_gpus - 1) / num_gpus;
  
  // Prefetch the arrays to all devices:
  #pragma omp parallel for num_threads(num_gpus)
  for(int i = 0; i < num_gpus; i++) {
     // === Duplicates the randoms array as read-only on all GPUs.
    cudaMemPrefetchAsync(randoms, n_rows * sizeof(int), i, 0);
    cudaMemPrefetchAsync(colors, n_rows * sizeof(int), i, 0);
    cudaMemPrefetchAsync(input, n_rows * sizeof(int), i, 0);

    // int color_begin = partitioned * i;
    // int color_end = partitioned * (i + 1);

    // === Pin portions of the memory of color/input array to each GPU.
    // a system containing multiple GPUs with peer-to-peer access enabled, 
    // where the data located on one GPU is occasionally accessed by other GPUs. 
    // In such scenarios, migrating data over to the other GPUs is not as 
    // important because the accesses are infrequent and the overhead of migration 
    // may be too high. But preventing faults can still help improve performance, 
    // and so having a mapping set up in advance is useful. 
    // if(i == num_gpus - 1) partitioned = n_rows - (partitioned * (num_gpus - 1));

    cudaMemAdvise(colors + partitioned * i, partitioned * sizeof(int), cudaMemAdviseSetPreferredLocation, i);
    cudaMemAdvise(input + partitioned * i, partitioned * sizeof(int), cudaMemAdviseSetPreferredLocation, i);

    // cudaMemAdvise(colors + partitioned * i, partitioned * sizeof(int), cudaMemAdviseSetAccessedBy, i);
    // cudaMemAdvise(input + partitioned * i, partitioned * sizeof(int), cudaMemAdviseSetAccessedBy, i);

    // cudaMemAdvise(colors, n_rows * sizeof(int), cudaMemAdviseSetAccessedBy, i);
    // cudaMemAdvise(input, n_rows * sizeof(int), cudaMemAdviseSetAccessedBy, i);

    // === Prefetch each portion ahead of time.
    // cudaMemPrefetchAsync(colors + partitioned * i, partitioned * sizeof(int), i, 0);
    // cudaMemPrefetchAsync(input + partitioned * i, partitioned * sizeof(int), i, 0);

  }

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

  int* indptr = g_indptr;
  int* indices = g_indices;
  float* data = g_data;
  
  int iteration = 0;
  while(iteration < 29) {
    t.begin();

    auto fn = [indptr, indices, data, colors, randoms, iteration] __host__ __device__(int const& vertex) {
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

        if(!colormax && !colormin) return vertex; // optimization
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
      
      auto input_begin  = input + chunk_size * i;
      auto input_end    = input + chunk_size * (i + 1);
      if(i == num_gpus - 1) input_end = input + n_rows;

      thrust::transform(
        thrust::cuda::par.on(infos[i].stream),
        input_begin,
        input_end,
        input_begin,
        fn
      );

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
  thrust::host_vector<int> out(colors, colors + n_rows);
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
  
  enable_peer_access();
  create_contexts();
  read_binary(inpath);

  int num_gpus = get_num_gpus();
  std::cout << "color | num_gpus: " << num_gpus << " vertices: " << n_rows << std::endl;

  int num_iters = 4;
  for(int i = 0; i < num_iters; i++)
    do_test();
  
  std::cout << "-----" << std::endl;
  return EXIT_SUCCESS;
}
