#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    // 在主机端分配页锁定内存。
    // 对于cuda在主机端的内存有分可分页内存(pageable memroy)和页锁定内存(page-lock或pinned)。
    // 页锁定后，主机的操作系统将不会对这块内存进行分页和交换操作，确保该内存始终驻留在物理内存中。
    // cuda可跟踪页锁定内存的物理地址，通过“直接内存访问(Direct Memory Access，DMA)”直接在主机和GPU之间复制数据，速率更快。
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    // 对应cudaMallocHost。
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  // 指向CPU端内存的指针。
  void* cpu_ptr_;
  // 指向GPU端内存的指针。
  void* gpu_ptr_;
  // 内存大小。
  size_t size_;
  // 用于标记正使用的数据是在哪一端。
  // 分{未初始化，在CPU端，在GPU端，同步的}四种状态。
  SyncedHead head_;
  // 用于标记是否已经存在由自己分配的数据，设置数据时要用到。
  bool own_cpu_data_;
  // 用于标记该CPU端的内存分配是否是通过GPU分配的页锁定内存。
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
