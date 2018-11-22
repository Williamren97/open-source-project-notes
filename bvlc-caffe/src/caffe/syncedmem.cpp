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
// �������ݵ�CPU�ˡ�
inline void SyncedMemory::to_cpu() {
  check_device();
  // ��Ҫ���ñ�ǵ�head_���жϵ�ǰʹ�õ�����������һ�ˡ�
  switch (head_) {
  // ��δ��ʼ�������CPU�˷���ҳ�����ڴ档
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  // �����ǰ��������GPU�ˣ��򿽱���ȥ��
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
  // ���������CPU�˻���ͬ���ģ���������
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

// �������ݵ�GPU�ˣ�������to_cpu���ơ�
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

// ��ȡCPU�����ݡ�
const void* SyncedMemory::cpu_data() {
  // �ȼ���豸������ǰʹ�õ�GPU�뿪ʼʱ�������õ�GPU�Ƿ�һ�¡�
  check_device();
  // �������ݵ�CPU�ˣ����������CPU�˵��򲻻���ʲô������
  to_cpu();
  return (const void*)cpu_ptr_;
}

// ����CPU�����ݡ�
void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  // ����ָ�벻��Ϊ�ա�
  CHECK(data);
  // �ж��Ƿ��Ѿ��������Լ����Ƶ����ݣ���������Ҫ���ͷŵ���
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  // ָ����ֱ�Ӹ������ģ��ڴ���������ƣ��ڲ���Ȩ�ͷţ�����Ϊfalse��
  own_cpu_data_ = false;
}

// ��ȡGPU�����ݡ�
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

// ����GPU�����ݣ�������set_cpu_data���ơ�
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

// ��ȡCPU�����ݵ�ָ�롣
// ��cpu_data()��һ��head_ = HEAD_AT_CPU��
// ��˼�ǵ���mutable_cpu_data()����ʱ���ⲿ����ֱ��ͨ�����ص�ָ�����޸����ݡ�
// ��cpu_data()���ص�ָ���ڴ��򲻿��޸ġ�
void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

// ��ȡGPU�����ݵ�ָ�룬������mutable_cpu_data()���ơ�
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
// �첽�����ݴ�CPU�˿�����GPU�ˡ�
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  // �첽����������
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  // ����ǣ���ʹ�����ݵ�ʱ����Ҫȷ���������Ѿ�������ϡ�
  head_ = SYNCED;
}
#endif

// ��鵱ǰʹ�õ�GPU��������õ�GPU�Ƿ�һ�¡�
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

