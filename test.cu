#include <cuda.h>
#include <nvToolsExt.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdlib>  // EXIT_SUCCESS

__global__ void fn_kernel(int n, int* x, int* y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int i = x[idx];
    int acc = 0;
    for (int ii = 0; ii < i; ii++) {
      acc += ii;
    }
    y[i] = (int)(acc % 2);
  }
}

template <typename type_t>
class physical_memory {
  using allocation_handle_t = CUmemGenericAllocationHandle;
  using allocation_properties_t = CUmemAllocationProp;

  allocation_handle_t alloc_handle;
  allocation_properties_t prop = {};
  std::size_t granularity;
  std::size_t padded_size;
  std::size_t size;
  unsigned long long flags;  // not used within CUDA

  int device_id;

  physical_memory(std::size_t _size, int _device_id)
      : size(_size), device_id(_device_id), flags(0) {
    // Set properties of the allocation to create.
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;

    cuMemGetAllocationGranularity(&granularity, &prop,
                                  CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    padded_size = (size + granularity - 1) / granularity;
    cuMemCreate(&alloc_handle, padded_size, &prop, flags);
  }
};

template <typename type_t>
class virtual_memory {
  using allocation_handle_t = CUmemGenericAllocationHandle;
  using allocation_properties_t = CUmemAllocationProp;

  type_t* ptr;               // pointer
  std::size_t size;          // padded size
  std::size_t alignment;     // alignment of reserved range
  type_t* addr;              // Fixed starting address range requested
  unsigned long long flags;  // not used within CUDA

  virtual_memory(std::size_t padded_size)
      : size(padded_size), alignment(0), addr(0), flags(0) {
    cuMemAddressReserve(&ptr, size, alignment, addr, flags);
  }
};

template <typename type_t>
class memory_mapper {
  const virtual_memory<type_t>& virt;

 public:
  memory_mapper(const virtual_memory<type_t>& virt_arg,
                const physical_memory<type_t>& phys_arg,
                const std::vector<int>& mapping_devices,
                unsigned int chunk)
      : virt(virt_arg) {
    const size_t size = phys_arg.padded_size;
    const size_t offset = size * chunk;
    cuMemMap(virt.ptr + offset, size, 0, phys_arg.alloc_handle, 0);

    std::vector<CUmemAccessDesc> access_descriptors(mapping_devices.size());

    for (unsigned int id = 0; id < mapping_devices.size(); id++) {
      access_descriptors[id].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access_descriptors[id].location.id = mapping_devices[id];
      access_descriptors[id].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    cuMemSetAccess(virt.ptr + offset, size, access_descriptors.data(),
                   access_descriptors.size());
  }

  ~memory_mapper() { cuMemUnmap(virt.ptr, virt.padded_size); }
};

void do_test(int num_arguments, char** argument_array) {
  // --
  // Create data

  int n = 400000;

  thrust::host_vector<int> h_input(n);
  thrust::host_vector<int> h_output(n);

  for (int i = 0; i < n; i++)
    h_input[i] = rand() % 1000000;
  thrust::fill(thrust::host, h_output.begin(), h_output.end(), -1);

  thrust::device_vector<int> input = h_input;
  thrust::device_vector<int> output_thrust = h_output;
  thrust::device_vector<int> output_kernel = h_output;

  // --
  // Setup data

  int num_gpus = 1;
  cudaGetDeviceCount(&num_gpus);

  // Peer access
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    for (int j = 0; j < num_gpus; j++) {
      if (i == j)
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
    cudaEvent_t event;
  };

  std::vector<gpu_info> infos;

  for (int i = 0; i < num_gpus; i++) {
    gpu_info info;
    cudaSetDevice(i);
    cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&info.event, cudaEventDisableTiming);
    infos.push_back(info);
  }

  // --
  // Thrust

  cudaSetDevice(0);

  auto fn = [=] __host__ __device__(int const& i) -> bool {
    int acc = 0;
    for (int ii = 0; ii < i; ii++)
      acc += ii;

    return (i + acc) % 2 == 0;
  };

  cudaDeviceSynchronize();
  nvtxRangePushA("thrust_work");
#pragma omp parallel for num_threads(num_gpus)
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);

    auto input_begin = input.begin() + chunk_size * i;
    auto input_end = input.begin() + chunk_size * (i + 1);
    auto output_begin = output_thrust.begin() + chunk_size * i;

    if (i == num_gpus - 1)
      input_end = input.end();

    thrust::copy_if(thrust::cuda::par.on(infos[i].stream), input_begin,
                    input_end, output_begin, fn);
    cudaEventRecord(infos[i].event, infos[i].stream);
  }

  for (int i = 0; i < num_gpus; i++)
    cudaStreamWaitEvent(master_stream, infos[i].event, 0);
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
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

  // for(int i = 0; i < num_gpus; i++) cudaStreamWaitEvent(master_stream,
  // infos[i].event, 0); for(int i = 0; i < num_gpus; i++) {cudaSetDevice(i);
  // cudaDeviceSynchronize();} nvtxRangePop();

  thrust::host_vector<int> ttmp = output_thrust;
  thrust::copy(ttmp.begin(), ttmp.begin() + 40,
               std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // thrust::host_vector<int> ktmp = output_kernel;
  // thrust::copy(ktmp.begin(), ktmp.begin() + 40,
  // std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;
}

int main(int argc, char** argv) {
  for (int i = 0; i < 10; i++)
    do_test(argc, argv);
  return EXIT_SUCCESS;
}