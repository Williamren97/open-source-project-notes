#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}
// 推送数据到CPU端。
inline void SyncedMemory::to_cpu() {
  check_device();
  // 需要先用标记的head_来判断当前使用的数据是在哪一端。
  switch (head_) {
  // 若未初始化，则给CPU端分配页锁定内存。
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  // 如果当前数据是在GPU端，则拷贝过去。
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  // 若本身就在CPU端或是同步的，则不做处理。
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

// 推送数据到GPU端，与上面to_cpu类似。
inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

// 获取CPU端数据。
const void* SyncedMemory::cpu_data() {
  // 先检查设备，看当前使用的GPU与开始时参数设置的GPU是否一致。
  check_device();
  // 推送数据到CPU端，若本身就在CPU端的则不会做什么操作。
  to_cpu();
  return (const void*)cpu_ptr_;
}

// 设置CPU端数据。
void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  // 输入指针不能为空。
  CHECK(data);
  // 判断是否已经存在由自己控制的数据，若有则需要先释放掉。
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  // 指针是直接赋过来的，内存由外面控制，内部无权释放，所以为false。
  own_cpu_data_ = false;
}

// 获取GPU端数据。
const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

// 设置GPU端数据，与上面set_cpu_data类似。
void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

// 获取CPU端数据的指针。
// 比cpu_data()多一句head_ = HEAD_AT_CPU。
// 意思是调用mutable_cpu_data()函数时，外部可以直接通过返回的指针来修改数据。
// 而cpu_data()返回的指针内存则不可修改。
void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

// 获取GPU端数据的指针，与上面mutable_cpu_data()类似。
void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
// 异步将数据从CPU端拷贝到GPU端。
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  // 异步拷贝函数。
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  // 做标记，在使用数据的时候需要确保该数据已经拷贝完毕。
  head_ = SYNCED;
}
#endif

// 检查当前使用的GPU与参数设置的GPU是否一致。
void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

