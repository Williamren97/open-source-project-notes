/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/session_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class StepStatsCollector;

// Executor runs a graph computation.
// Example:
//   Graph* graph = ...;
//      ... construct graph ...
//   Executor* executor;
//   TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
//   Rendezvous* rendezvous = NewNaiveRendezvous();
//   TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
//   TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
//   TF_CHECK_OK(rendezvous->Recv("output", &output_tensor));
//   ... ...
//
// Multiple threads can call Executor::Run concurrently.
// Executor用于执行计算图的运算，运行函数是Run。
// 多个线程可以同时调用Executor::Run。
class Executor {
 public:
  virtual ~Executor() {}

  // RunAsync() executes the graph computation. "done" is run when the
  // graph computation completes. If any error happens during the
  // computation, "done" is run and the error is passed to "done".
  //
  // RunAsync() is given a few arguments in Args. The caller must
  // ensure objects passed in Args (rendezvous, stats_collector, etc.)
  // are alive at least until done is invoked. All pointers to the
  // argument objects can be nullptr.
  //
  // "step_id" is a process-wide unique identifier for the step being
  // run. Executors on different devices may receive the same step_id
  // in the case that a step runs Ops on more than one device. The
  // step_id is used for tracking resource usage of a given step.
  //
  // RunAsync() uses the given "rendezvous", if not null, as the
  // mechanism to communicate inputs and outputs of the underlying
  // graph computation.
  // 
  // RunAsync() calls "stats_collector", if not null, to keep track of
  // stats. This allows us to collect statistics and traces on demand.
  //
  // RunAsync() is provided a "call_frame", if the executor is used
  // for executing a function, is used to pass arguments and return
  // values between the caller and the callee.
  //
  // RunAsync() uses "cancellation_manager", if not nullptr, to
  // register callbacks that should be called if the graph computation
  // is canceled. Note that the callbacks merely unblock any
  // long-running computation, and a canceled step will terminate by
  // returning/calling the DoneCallback as usual.
  //
  // RunAsync() dispatches closures to "runner". Typically, "runner"
  // is backed up by a bounded threadpool.
  struct Args {
    // 运行步骤中跨进程的唯一标识符。在一个步骤在多个设备上运行操作的情况下，
    // 不同设备上的执行器可以接收到相同的step_id。
    // step_id用于跟踪给定步骤的资源使用情况。
    int64 step_id = 0;
    // 如果不为空，则作为底层graph计算中输入和输出的通信机制。
    Rendezvous* rendezvous = nullptr;
    // 如果不是null，则跟踪统计数据stats。这使我们能够根据需要收集统计数据和跟踪。
    StepStatsCollector* stats_collector = nullptr;
    // 如果executor用于执行函数，则用于在caller和callee之间传递参数和返回值。
    CallFrameInterface* call_frame = nullptr;
    // 如果不是nullptr，则该注册了的回调函数应在取消graph计算时调用。
    // 注意，回调只是解除了任何长时间运行的计算，取消的步骤将通过像
    // 往常一样返回/调用DoneCallback而终止。
    CancellationManager* cancellation_manager = nullptr;
    SessionState* session_state = nullptr;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    CollectiveExecutor* collective_executor = nullptr;

    // If true, calls Sync() on the device.
    bool sync_on_finish = false;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner;
    Runner runner = nullptr;

    // A callback that is invoked each time a node has finished executing.
    typedef std::function<Status(const string& node_name, const int output_slot,
                                 const Tensor* tensor, const bool is_ref,
                                 OpKernelContext* ctx)>
        NodeOutputsCallback;
  };
  typedef std::function<void(const Status&)> DoneCallback;
  // 执行计算图的计算：
  // args为一些输入参数，在done调用前不能释放；
  // done会在计算完成后调用，由外部实现。
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;

  // Synchronous wrapper for RunAsync().
  Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

// Creates an Executor that computes the given "graph".
//
// If successful, returns the constructed executor in "*executor". Otherwise,
// returns an error status.
//
// "params" provides a set of context for the executor. We expect that
// different context would provide different implementations.
struct LocalExecutorParams {
  Device* device;

  // The library runtime support.
  FunctionLibraryRuntime* function_library = nullptr;

  // create_kernel returns an instance of op kernel based on NodeDef.
  // delete_kernel is called for every kernel used by the executor
  // when the executor is deleted.
  // 基于NodeDef返回一个op kernel的实例
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel;
  // 在删除executor时，将为executor使用的每个内核调用delete_kernel。
  std::function<void(OpKernel*)> delete_kernel;
};
// 创建用于计算graph的executor
::tensorflow::Status NewLocalExecutor(const LocalExecutorParams& params,
                                      std::unique_ptr<const Graph> graph,
                                      Executor** executor);

// A class to help run multiple executors in parallel and wait until
// all of them are complete.
//
// ExecutorBarrier deletes itself after the function returned by Get()
// is called.
//
// 用于帮助并行运行多个执行器，并等待所有执行器完成计算。
class ExecutorBarrier {
 public:
  typedef std::function<void(const Status&)> StatusCallback;

  // Create an ExecutorBarrier for 'num' different executors.
  //
  // 'r' is the shared Rendezvous object that is used to communicate
  // state.  If any of the executors experiences an error, the
  // rendezvous object will be aborted exactly once.
  //
  // 'done' is called after the last executor completes, and
  // ExecutorBarrier is deleted.
  //
  // num：执行器的数量。
  // r：共享的Rendezvous对象，用于state的交互通讯。
  // done：回调函数StatusCallback，在所有执行器都执行完时，该函数会被调用，
  //       同时ExecutorBarrier也会被删除
  ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}

  ~ExecutorBarrier() {}

  // Returns a closure that Executors must call when they are done
  // computing, passing the status of their execution as an argument.
  // 返回一个执行器在执行完毕之后必须调用的函数闭包，
  // 执行器会使用它们结束时的状态作为执行闭包的参数
  StatusCallback Get() {
    // std::bind预先参数this绑定到已有的函数WhenDone，
    // 产生一个新的可调用的实体std::function
    return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
  }

 private:
  Rendezvous* rendez_ = nullptr;
  StatusCallback done_cb_ = nullptr;

  mutable mutex mu_;
  // 还剩几个执行器未执行完
  int pending_ GUARDED_BY(mu_) = 0;
  Status status_ GUARDED_BY(mu_);

  void WhenDone(const Status& s) {
    bool error = false;
    Rendezvous* error_rendez = nullptr;
    StatusCallback done = nullptr;
    Status status;
    {
      mutex_lock l(mu_);
      // If we are the first error encountered, mark the status
      // appropriately and later trigger an abort of the Rendezvous
      // object by this thread only.
      if (status_.ok() && !s.ok()) {
        error = true;
        error_rendez = rendez_;
        error_rendez->Ref();
        status_ = s;
      }

      // If this is the last call to WhenDone, call the final callback
      // below.
      if (--pending_ == 0) {
        CHECK(done_cb_ != nullptr);
        std::swap(done, done_cb_);
      }

      if (!status_.ok()) {
        status = status_;
      }
    }

    if (error) {
      error_rendez->StartAbort(status);
      error_rendez->Unref();
    }
    if (done != nullptr) {
      delete this;
      done(status);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorBarrier);
};

// A few helpers to facilitate create/delete kernels.

// Creates a kernel based on "ndef" on device "device". The kernel can
// access the functions in the "flib". The caller takes ownership of
// returned "*kernel".
// 在设备“device”上创建基于“ndef”的内核。
// 内核可以访问“flib”中的函数。调用方获得返回的“*kernel”的所有权。
Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel);

// Deletes "kernel" returned by CreateKernel.
void DeleteNonCachedKernel(OpKernel* kernel);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_EXECUTOR_H_
