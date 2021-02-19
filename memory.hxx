#pragma once

static size_t round_up(std::size_t x, std::size_t y) { 
  return ((x + y - 1) / y) * y; 
}

#include <cuda.h>
#include <vector>

template <typename type_t>
struct physical_memory {
  using allocation_handle_t = CUmemGenericAllocationHandle;
  using allocation_properties_t = CUmemAllocationProp;

  std::vector<allocation_handle_t> alloc_handle;
  allocation_properties_t prop = {};
  std::size_t granularity;
  std::size_t padded_size;
  std::size_t stripe_size;
  std::size_t size;
  unsigned long long flags;  // not used within CUDA

  std::vector<int> resident_devices;

  physical_memory(std::size_t _size, 
                  const std::vector<int> _resident_devices)
      : size(_size), 
        resident_devices(_resident_devices), 
        flags(0), 
        granularity(0) {
    // Set properties of the allocation to create.
    // The following properties will create a pinned memory, local
    // to the device (GPU).
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    // Find the minimum granularity needed for the resident devices.
    for (std::size_t idx = 0; idx < resident_devices.size(); idx++) {
        std::size_t _granularity = 0;
        // get the minnimum granularity for residentDevices[idx]
        prop.location.id = resident_devices[idx];
        cuMemGetAllocationGranularity(&_granularity, &prop,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (granularity < _granularity)
            granularity = _granularity;
    }

    // Round up the size such that it can evenly split into a stripe size
    // that meets the granularity requirement. padded_size = N * GPUs *
    // granularity, since each of the piece of the allocation will be
    // N * granularity and the granularity applies to each stripe_size
    // piece of the allocation.
    // IMPORTANT! Use padded_size for free().
    padded_size = round_up(size, resident_devices.size() * granularity);
    stripe_size = padded_size / resident_devices.size();

    // Create the backings on each GPU.
    alloc_handle.resize(resident_devices.size());
    for (std::size_t idx = 0; idx < resident_devices.size(); idx++) {
        prop.location.id = resident_devices[idx];
        cuMemCreate(&alloc_handle[idx], stripe_size, &prop, flags);
    }
  }

  ~physical_memory() {
      for (std::size_t idx = 0; idx < resident_devices.size(); idx++)
        cuMemRelease(alloc_handle[idx]);
  }
};

template <typename type_t>
struct virtual_memory {
  type_t* ptr;               // pointer
  std::size_t size;          // padded size
  std::size_t alignment;     // alignment of reserved range
  type_t* addr;              // Fixed starting address range requested
  unsigned long long flags;  // not used within CUDA

  virtual_memory(std::size_t padded_size)
      : size(padded_size), alignment(0), addr(0), flags(0) {
    cuMemAddressReserve((CUdeviceptr*)&ptr, size, alignment, (CUdeviceptr)addr, flags);
  }

  ~virtual_memory() {
    cuMemAddressFree((CUdeviceptr)ptr, size);
  }
};

template <typename type_t>
class memory_mapper {
  const virtual_memory<type_t>& virt;

 public:
  memory_mapper(const virtual_memory<type_t>& virt_arg,
                const physical_memory<type_t>& phys_arg,
                const std::vector<int>& mapping_devices)
      : virt(virt_arg) {
    const size_t size = phys_arg.padded_size;
    const size_t stripe_size = phys_arg.stripe_size;
    for(std::size_t idx = 0; idx < phys_arg.resident_devices.size(); idx++) {
        cuMemMap((CUdeviceptr)virt.ptr + (stripe_size * idx), 
                 stripe_size, 0, phys_arg.alloc_handle[idx], 0);
    }

    std::vector<CUmemAccessDesc> access_descriptors(mapping_devices.size());

    for (std::size_t idx = 0; idx < mapping_devices.size(); idx++) {
      access_descriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access_descriptors[idx].location.id = mapping_devices[idx];

      // Enable READ & WRITE.
      access_descriptors[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    cuMemSetAccess((CUdeviceptr)virt.ptr, size, access_descriptors.data(),
                   access_descriptors.size());
  }

  ~memory_mapper() { cuMemUnmap((CUdeviceptr)virt.ptr, virt.size); }

  type_t* data() { return (type_t*)virt.ptr; }
};
