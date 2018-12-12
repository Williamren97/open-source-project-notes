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

// A Graph describes a set of computations that are to be
// performed, as well as the dependencies between those
// computations. The basic model is a DAG (directed acyclic graph) with
// * internal nodes representing computational operations to be performed;
// * edges represent dependencies, indicating the target may only be
//   executed once the source has completed; and
// * predefined "source" (start) and "sink" (finish) nodes -- the source
//   should be the only node that doesn't depend on anything, and the sink
//   should be the only node that nothing depends on.
//
// Note: Node ids are intended to be relatively dense in the
// 0..max_id range, but there may be gaps since ids won't be reused.
//
// Note: Some dependencies between operations are due to one operation
// consuming the output of another. In fact operations can produce
// multiple outputs and consume multiple inputs, and some
// optimizations will care about which specific outputs are connected
// to which specific inputs.  We therefore represent data dependency
// between output O of layer A and input I of layer B using
// "input index" and "output index" labels per edge.

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_H_

#include <functional>
#include <string>
#include <vector>
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Edge;
class EdgeSetTest;
class Graph;
class GraphDef;
class Node;
class VersionDef;
class WhileContext;

class NeighborIter;    // Declared below
class NodeIter;        // Declared below
class NodeProperties;  // Defined in .cc

// 节点
class Node {
 public:
  string DebugString() const;
  int id() const { return id_; }
  int cost_id() const { return cost_id_; }
  const string& name() const;
  const string& type_string() const;

  // def() provides the NodeDef the user supplied, but the specifics
  // of this Node may have changed due to placement, optimization, etc.
  // In particular:
  // * def().name() will match name();
  // * def().op() will match type_string() and op_def().name();
  // * def().input() is not reliable, use "in_edges()" below instead;
  // * def().device() is the "user's requested device" and may not match
  //   the actual assigned device, see assigned_device_name() below;
  // * def().attr() is authoritative.
  // TODO(irving): Replace with NodeInfo.
  // NodeDef和OpDef均继承于protobuf::Message
  // 提供用户提供的节点定义，但由于位置、优化等原因，该节点的细节可能已经更改。
  const NodeDef& def() const;
  const OpDef& op_def() const;

  // input and output types
  int32 num_inputs() const;
  DataType input_type(int32 i) const;
  const DataTypeVector& input_types() const;

  int32 num_outputs() const;
  DataType output_type(int32 o) const;
  const DataTypeVector& output_types() const;

  // The device requested by the user.  For the actual assigned device,
  // use assigned_device_name() below.
  // 用户请求的设备。对于实际分配的设备，请使用下面的assigned_device_name()。
  const string& requested_device() const;

  // This changes the user requested device but not necessarily the device that
  // on which the operation will run.
  // 这将更改用户请求的设备，但不一定更改真正运行op的设备。
  void set_requested_device(const string& device);

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move assigned_device_name outside of Node into a
  // NodeId->DeviceName map.
  const string& assigned_device_name() const;
  void set_assigned_device_name(const string& device_name);
  bool has_assigned_device_name() const {
    return assigned_device_name_index_ > 0;
  }
  int assigned_device_name_index() const { return assigned_device_name_index_; }
  void set_assigned_device_name_index(int index);

  // Read only access to attributes
  // 获取属性信息，输出的类对象是根据node的定义def()来构建的
  AttrSlice attrs() const;

  // Inputs requested by the NodeDef.  For the actual inputs, use in_edges.
  // 由NodeDef请求的输入。对于实际输入，使用in_edge
  const protobuf::RepeatedPtrField<string>& requested_inputs() const;

  // Get the neighboring nodes via edges either in or out of this node.  This
  // includes control edges.
  // 通过该节点的输入或输出边去获取其相邻的节点。这包括控制边。
  gtl::iterator_range<NeighborIter> in_nodes() const;
  gtl::iterator_range<NeighborIter> out_nodes() const;
  const EdgeSet& in_edges() const { return in_edges_; }
  const EdgeSet& out_edges() const { return out_edges_; }

  // Node type helpers.
  bool IsSource() const { return id() == 0; }
  bool IsSink() const { return id() == 1; }
  // Anything other than the special Source & Sink nodes.
  // 除了特殊的源Source和接收Sink节点以外的任何节点。
  bool IsOp() const { return id() > 1; }

  // Node class helpers
  // 判断节点类型
  bool IsSwitch() const { return class_ == NC_SWITCH; }
  bool IsMerge() const { return class_ == NC_MERGE; }
  bool IsEnter() const { return class_ == NC_ENTER; }
  bool IsExit() const { return class_ == NC_EXIT; }
  bool IsNextIteration() const { return class_ == NC_NEXT_ITERATION; }
  bool IsLoopCond() const { return class_ == NC_LOOP_COND; }
  bool IsControlTrigger() const { return class_ == NC_CONTROL_TRIGGER; }
  bool IsSend() const { return class_ == NC_SEND || class_ == NC_HOST_SEND; }
  bool IsRecv() const { return class_ == NC_RECV || class_ == NC_HOST_RECV; }
  bool IsConstant() const { return class_ == NC_CONSTANT; }
  bool IsVariable() const { return class_ == NC_VARIABLE; }
  bool IsIdentity() const { return class_ == NC_IDENTITY; }
  bool IsGetSessionHandle() const { return class_ == NC_GET_SESSION_HANDLE; }
  bool IsGetSessionTensor() const { return class_ == NC_GET_SESSION_TENSOR; }
  bool IsDeleteSessionTensor() const {
    return class_ == NC_DELETE_SESSION_TENSOR;
  }
  bool IsControlFlow() const {
    return (class_ != NC_OTHER) &&  // Fast path
           (IsSwitch() || IsMerge() || IsEnter() || IsExit() ||
            IsNextIteration());
  }
  bool IsHostSend() const { return class_ == NC_HOST_SEND; }
  bool IsHostRecv() const { return class_ == NC_HOST_RECV; }
  bool IsScopedAllocator() const { return class_ == NC_SCOPED_ALLOCATOR; }
  bool IsCollective() const { return class_ == NC_COLLECTIVE; }

  bool IsMetadata() const { return class_ == NC_METADATA; }

  template <typename T>
  void AddAttr(const string& name, const T& val) {
    SetAttrValue(val, AddAttrHelper(name));
  }

  void ClearAttr(const string& name);

  // Returns into '*e' the edge connecting to the 'idx' input of this Node.
  // 返回该节点的第idx个输入边e。
  Status input_edge(int idx, const Edge** e) const;

  // Returns into '*edges' the input data edges of this Node, indexed by input
  // number. Does not return control edges.
  // 返回该节点的数据输入边edges。不返回控制边(control edges)
  Status input_edges(std::vector<const Edge*>* edges) const;

  // Returns into '*n' the node that has an output connected to the
  // 'idx' input of this Node.
  // 返回该节点的第idx个输入节点。
  Status input_node(int idx, const Node** n) const;
  Status input_node(int idx, Node** n) const;

  // WhileContext：关于一个while loop的信息，每个用户定义的
  //               while loop都会有其相关的WhileContext.
  WhileContext* while_ctx() const { return while_ctx_; }
  void set_while_ctx(WhileContext* while_ctx) {
    DCHECK(IsExit());
    DCHECK(while_ctx_ == nullptr);
    while_ctx_ = while_ctx;
  }

 private:
  friend class Graph;
  Node();

  NodeProperties* properties() const { return props_.get(); }

  void Initialize(int id, int cost_id, std::shared_ptr<NodeProperties> props);

  // Releases memory from props_, in addition to restoring *this to its
  // uninitialized state.
  void Clear();

  // Make a copy of the Node's props_ if props_ is shared with
  // other nodes. This must be called before mutating properties,
  // e.g. in AddAttr.
  // 如果props_是与其他节点共享的，则对该节点的props_做个备份。
  void MaybeCopyOnWrite();

  // 返回该节点属性中索引为name的AttrValue。
  // 该函数用于上面定义的AddAttr。
  AttrValue* AddAttrHelper(const string& name);

  // A set of mutually exclusive classes for different kinds of nodes,
  // class_ is initialized in the Node::Initialize routine based on the
  // node's type_string().
  // 对于不同类型的节点，class_是一组互斥的类，
  // 在Node::Initialize例程中根据节点的type_string()初始化class_。
  enum NodeClass {
    NC_UNINITIALIZED,
    NC_SWITCH,
    NC_MERGE,
    NC_ENTER,
    NC_EXIT,
    NC_NEXT_ITERATION,
    NC_LOOP_COND,
    NC_CONTROL_TRIGGER,
    NC_SEND,
    NC_HOST_SEND,
    NC_RECV,
    NC_HOST_RECV,
    NC_CONSTANT,
    NC_VARIABLE,
    NC_IDENTITY,
    NC_GET_SESSION_HANDLE,
    NC_GET_SESSION_TENSOR,
    NC_DELETE_SESSION_TENSOR,
    NC_METADATA,
    NC_SCOPED_ALLOCATOR,
    NC_COLLECTIVE,
    NC_OTHER  // Not a special kind of node
  };

  // 用字符串与NodeClass中的枚举进行对应，如Switch对应NC_SWITCH。
  static const std::unordered_map<string, NodeClass>& kNodeClassTable;

  // 以字符串根据上面的kNodeClassTable找到对应的NodeClass。
  static NodeClass GetNodeClassForOp(const string& ts);

  int id_;       // -1 until Initialize() is called
  int cost_id_;  // -1 if there is no corresponding cost accounting node
  NodeClass class_;

  EdgeSet in_edges_;
  EdgeSet out_edges_;

  // NOTE(skyewm): inheriting from core::RefCounted may have a slight
  // performance benefit over using shared_ptr, at the cost of manual ref
  // counting
  std::shared_ptr<NodeProperties> props_;

  // Index within Graph::device_names_ of the name of device assigned
  // to perform this computation.
  // 在Graph::device_names_中指定执行此计算的设备名称的索引。
  int assigned_device_name_index_;

  // A back-pointer to the Graph that owns this node.  Currently, this exists
  // solely to allow Node::[set_]assigned_device_name() to work. However, if all
  // callers of Node::[set_]assigned_device_name() are modified to use the
  // equivalent methods defined directly on Graph, then we can remove this
  // field and reclaim that memory.
  // 一个反向指针，指向拥有该节点的图形。
  // 目前，它只允许Node::[set_]assigned_device_name()工作。
  // 但如果修改了Node::[set_]assigned_device_name()的所有调用者，
  // 以使用直接在图上定义的等效方法，那么我们可以删除该字段并回收该内存。
  Graph* graph_;

  // Set if this is an exit node of a while loop with an associated
  // WhileContext. Otherwise null. (This is only set for exit nodes because
  // they're the first nodes of a loop encountered while creating the gradient
  // graph. Exit nodes that are part of while loop gradient graphs will not have
  // this set.)
  // 如果这是一个while loop的出口节点，则设置它与关联的WhileContext，否则为NULL。
  // (这只用于出口节点，因为它们是创建梯度图时的循环中第一个节点。
  // 而while loop梯度图的退出节点不会有此集合)
  WhileContext* while_ctx_;

  TF_DISALLOW_COPY_AND_ASSIGN(Node);
};

// Represents an input of a node, i.e., the `index`-th input to `node`.
// 表示一个节点的一个输入
struct InputTensor {
  // 服务于该节点
  const Node* node;
  // 为该节点的第index个输入
  int index;

  InputTensor(const Node* n, int i) : node(n), index(i) {}
  InputTensor() : node(nullptr), index(0) {}

  // Returns true if this InputTensor is identical to 'other'. Nodes are
  // compared using pointer equality.
  bool operator==(const InputTensor& other) const;

  // A hash function for InputTensors. Nodes are hashed based on their pointer
  // value.
  struct Hash {
    uint64 operator()(InputTensor const& s) const;
  };
};

// Represents an output of a node, i.e., the `index`-th output of `node`. Note
// that a single `OutputTensor` can correspond to multiple `Edge`s if the output
// is consumed by multiple destination nodes.
// 一个节点的一个输出。
// 注意如果该输出被多个目标节点使用，那么一个OutputTensor可以对应多个Edge。
struct OutputTensor {
  // 服务于该节点
  const Node* node;
  // 为该节点的第index个输出
  int index;

  OutputTensor(const Node* n, int i) : node(n), index(i) {}
  OutputTensor() : node(nullptr), index(0) {}

  // Returns true if this OutputTensor is identical to 'other'. Nodes are
  // compared using pointer equality.
  bool operator==(const OutputTensor& other) const;

  // A hash function for OutputTensors. Nodes are hashed based on their pointer
  // value.
  struct Hash {
    uint64 operator()(OutputTensor const& s) const;
  };
};

// 计算图中的边
class Edge {
 public:
  // 返回该Edge连接的输入输出节点
  Node* src() const { return src_; }
  Node* dst() const { return dst_; }
  int id() const { return id_; }

  // Return the index of the source output that produces the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  // 返回产生由这条边携带的数据的源输出的索引。特殊值kControlSlot用于控制依赖项。
  int src_output() const { return src_output_; }

  // Return the index of the destination input that consumes the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  // 返回使用此边携带的数据的目标输入的索引。
  int dst_input() const { return dst_input_; }

  // Return true iff this is an edge that indicates a control-flow
  // (as opposed to a data-flow) dependency.
  // 如果这是一条表示控件流(而不是数据流)依赖关系的边，则返回true。
  bool IsControlEdge() const;

  string DebugString() const;

 private:
  Edge() {}

  friend class EdgeSetTest;
  friend class Graph;
  Node* src_;
  Node* dst_;
  int id_;
  int src_output_;
  int dst_input_;
};

// Allows for iteration of the edges of a Graph, by iterating the underlying
// Graph.edges_ vector while skipping over null entries.
// 允许迭代graph的edges，通过迭代底层Graph.edges_向量并同时跳过空条目。
class GraphEdgesIterable {
 private:
  const std::vector<Edge*>& edges_;

 public:
  // 显式构造：https://blog.csdn.net/starlee/article/details/1331268
  // 防止隐式构造造成错误
  explicit GraphEdgesIterable(const std::vector<Edge*>& edges)
      : edges_(edges) {}

  typedef Edge* value_type;

  class const_iterator {
   private:
    // The underlying iterator.
    std::vector<value_type>::const_iterator iter_;

    // The end of the underlying iterator.
    std::vector<value_type>::const_iterator end_;

    // Advances iter_ until it reaches a non-null item, or reaches the end.
    void apply_filter() {
      while (iter_ != end_ && *iter_ == nullptr) {
        ++iter_;
      }
    }

   public:
    const_iterator(std::vector<value_type>::const_iterator iter,
                   std::vector<value_type>::const_iterator end)
        : iter_(iter), end_(end) {
      apply_filter();
    }

    bool operator==(const const_iterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const const_iterator& other) const {
      return iter_ != other.iter_;
    }

    // This is the prefix increment operator (++x), which is the operator
    // used by C++ range iteration (for (x : y) ...).  We intentionally do not
    // provide a postfix increment operator.
    const_iterator& operator++() {
      ++iter_;
      apply_filter();
      return *this;
    }

    value_type operator*() { return *iter_; }
  };

  const_iterator begin() {
    return const_iterator(edges_.begin(), edges_.end());
  }
  const_iterator end() { return const_iterator(edges_.end(), edges_.end()); }
};

// Thread compatible but not thread safe.
// 线程兼容但不是线程安全的。
// 线程兼容类不是线程安全的，但是可以通过正确使用同步而在并发环境中安全地使用.
// https://baike.baidu.com/item/%E7%BA%BF%E7%A8%8B%E5%AE%89%E5%85%A8/9747724
class Graph {
 public:
  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in registry. `registry`s lifetime must be at
  // least that of the constructed graph's.
  //
  // 以一个源SOURCE、一个接收SINK节点和一条源节点到接收节点的边，去构建一个计算图。
  // 该计算图可以保存registry中找到的ops。
  // registry的生命周期必须长于构造的计算图的生命周期。
  explicit Graph(const OpRegistryInterface* registry);

  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in `flib_def`. Unlike the constructor taking
  // an OpRegistryInterface, this constructor copies the function definitions in
  // `flib_def` so its lifetime may be shorter than that of the graph's. The
  // OpRegistryInterface backing `flib_def` must still have the lifetime of the
  // graph though.
  //
  // 该计算图可以保存在flib_def中找到的ops。与使用OpRegistryInterface的构造函数不同，
  // 这个构造函数复制了flib_def中的函数定义，因此它的生命周期可能比计算图的生命周期短。
  // 但是，支持“flib_def”的OpRegistryInterface必须仍然具有计算图的生命周期。
  explicit Graph(const FunctionLibraryDefinition& flib_def);

  ~Graph();

  static const int kControlSlot;

  // The GraphDef version range of this graph (see graph.proto).
  // 该计算图的GraphDef的版本范围
  const VersionDef& versions() const;
  void set_versions(const VersionDef& versions);

  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  // 向计算图中添加一个新节点，并返回它。推断节点的Op和输入/输出类型。
  // *this拥有返回的实例。若返回nullptr，则在*status上设置错误信息。
  Node* AddNode(const NodeDef& node_def, Status* status);

  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  // 拷贝节点。该节点可能属于其他计算图，拷贝生成一个新的节点返回。
  // 不拷贝任何edges。*this拥有返回的实例。
  Node* CopyNode(const Node* node);

  // Removes a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  // 从计算图中移除一个节点，包括它的所有输入输出edges。
  // 在调用该节点后，不应再访问*node。
  void RemoveNode(Node* node);

  // Adds an edge that connects the xth output of `source` to the yth input of
  // `dest` and returns it. Does not update dest's NodeDef.
  // 添加一条边，将节点source的第x个输出连接到节点dest的第y个输入，并返回该边。
  // 不更新dest的节点定义。
  const Edge* AddEdge(Node* source, int x, Node* dest, int y);

  // Adds a control edge (no data flows along this edge) that connects `source`
  // to `dest`. If `dest`s NodeDef is missing the corresponding control input,
  // adds the control input.
  //
  // If such a control edge already exists and `allow_duplicates` is false, no
  // edge is added and the function returns nullptr. Otherwise the edge is
  // unconditionally created and returned. The NodeDef is not updated if
  // `allow_duplicates` is true.
  // TODO(skyewm): // TODO(skyewm): allow_duplicates is needed only by
  // graph_partition.cc. Figure out if we can do away with it.
  // 
  // 添加连接source和dest的control edge(该edge没有数据流)。
  // 如果dest的NodeDef缺少相应的控件(control)输入，则添加控件输入。
  //
  // 如果这样的control edge已经存在，并且allow_duplicate为false，
  // 则不添加任何edge，函数返回nullptr。否则，将无条件创建并返回edge。
  // 如果allow_duplicate为true，则不更新节点定义。
  const Edge* AddControlEdge(Node* source, Node* dest,
                             bool allow_duplicates = false);

  // Removes edge from the graph. Does not update the destination node's
  // NodeDef.
  // REQUIRES: The edge must exist.
  // 从图形中移除edge。不更新目标节点的NodeDef。
  void RemoveEdge(const Edge* edge);

  // Removes control edge `edge` from the graph. Note that this also updates
  // the corresponding NodeDef to reflect the change.
  // REQUIRES: The control edge must exist.
  // 从图形中移除control edge e。需要注意的是，这会更新相应的NodeDef以反映该更改。
  void RemoveControlEdge(const Edge* e);
  // Updates the input to a node.  The existing edge to `dst` is removed and an
  // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
  // is also updated.
  // 更新节点的输入。删除到dst的现有edge，并创建从new_src到dst的edge。
  // 与dst关联的NodeDef也会更新。
  Status UpdateEdge(Node* new_src, int new_src_index, Node* dst, int dst_index);

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  // 将fdef_lib中的函数和梯度定义添加到图的op注册表中。
  // 忽略重复函数，如果导入的函数与现有函数或op不同，但名字却相同的，则返回错误状态。
  Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib);

  // The number of live nodes in the graph.
  //
  // Because nodes can be removed from the graph, num_nodes() is often
  // smaller than num_node_ids(). If one needs to create an array of
  // nodes indexed by node ids, num_node_ids() should be used as the
  // array's size.
  // 
  // 计算图中活动节点的数量。
  // 因为节点可以从图中删除，所以num_nodes()通常小于num_node_ids()。
  // 如果需要创建按节点id索引的节点数组，则应该使用num_node_ids()作为数组的大小。
  int num_nodes() const { return num_nodes_; }

  // The number of live nodes in the graph, excluding the Source and Sink nodes.
  // 计算图图中活动节点的数量，不包括源节Source点和接收Sink节点
  int num_op_nodes() const {
    DCHECK_GE(num_nodes_, 2);
    return num_nodes_ - 2;
  }

  // The number of live edges in the graph.
  //
  // Because edges can be removed from the graph, num_edges() is often
  // smaller than num_edge_ids(). If one needs to create an array of
  // edges indexed by edge ids, num_edge_ids() should be used as the
  // array's size.
  // 与num_nodes()类似，针对edges。
  int num_edges() const { return num_edges_; }

  // Serialize the nodes starting at `from_node_id` to a GraphDef.
  // 将从from_node_id开始的节点序列化为GraphDef。
  void ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const;

  // Serialize to a GraphDef.
  // 将所有节点序列化为GraphDef，直接调用ToGraphDefSubRange(graph_def, 0);
  void ToGraphDef(GraphDef* graph_def) const;

  // This version can be called from debugger to inspect the graph content.
  // Use the previous version outside debug context for efficiency reasons.
  //
  // Note: We do not expose a DebugString() API, since GraphDef.DebugString() is
  // not defined in some TensorFlow builds.
  // 检查图形内容.
  GraphDef ToGraphDefDebug() const;

  // Generate new node name with the specified prefix that is unique
  // across this graph.
  // 生成具有指定前缀的新节点名，该前缀在此图中是惟一的。
  string NewName(StringPiece prefix);

  // Access to the list of all nodes.  Example usage:
  //   for (Node* node : graph.nodes()) { ... }
  // 访问所有节点的列表
  gtl::iterator_range<NodeIter> nodes() const;

  // Access to the list of all nodes, excluding the Source and Sink nodes.
  // 访问所有节点的列表，除了源Source和接收Sink节点
  gtl::iterator_range<NodeIter> op_nodes() const;

  // Returns one more than the maximum id assigned to any node.
  // 返回比分配给任何节点的最大id多一的值。
  int num_node_ids() const { return nodes_.size(); }

  // Returns the node associated with an id, or nullptr if no node
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  // 根据id号找节点
  Node* FindNodeId(int id) const { return nodes_[id]; }

  // Returns one more than the maximum id assigned to any edge.
  // 返回比分配给任何edges的最大id多一的值。
  int num_edge_ids() const { return edges_.size(); }

  // Returns the Edge associated with an id, or nullptr if no edge
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  // 根据id号找edge
  const Edge* FindEdgeId(int id) const { return edges_[id]; }

  // Access to the set of all edges.  Example usage:
  //   for (const Edge* e : graph.edges()) { ... }
  // 访问所有edges。
  GraphEdgesIterable edges() const { return GraphEdgesIterable(edges_); }

  // The pre-defined nodes.
  // 预定义的节点，在构建计算图时，最先添加的就是源和接收节点，id分别为0和1.
  enum { kSourceId = 0, kSinkId = 1 };
  Node* source_node() const { return FindNodeId(kSourceId); }
  Node* sink_node() const { return FindNodeId(kSinkId); }

  const OpRegistryInterface* op_registry() const { return &ops_; }
  const FunctionLibraryDefinition& flib_def() const { return ops_; }

  void CheckDeviceNameIndex(int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, static_cast<int>(device_names_.size()));
  }

  // 确保device_name出现在设备名称表中，并返回该设备名称的索引。
  // 该索引是稳定的，可以在调用Node::set_assigned_device_name_index()时使用。
  int InternDeviceName(const string& device_name);

  const string& get_assigned_device_name(const Node& node) const {
    return device_names_[node.assigned_device_name_index()];
  }

  void set_assigned_device_name_index(Node* node, int device_name_index) {
    CheckDeviceNameIndex(device_name_index);
    node->assigned_device_name_index_ = device_name_index;
  }

  void set_assigned_device_name(Node* node, const string& device_name) {
    node->assigned_device_name_index_ = InternDeviceName(device_name);
  }

  // Returns OK if `node` is non-null and belongs to this graph
  Status IsValidNode(const Node* node) const;

  // Returns OK if IsValidNode(`node`) and `idx` is less than
  // node->num_outputs()
  Status IsValidOutputTensor(const Node* node, int idx) const;

  // Returns OK if IsValidNode(`node`) and `idx` is less than
  // node->num_inputs()
  Status IsValidInputTensor(const Node* node, int idx) const;

  // Create and return a new WhileContext owned by this graph. This is called
  // when a new while loop is created. `frame_name` must be unique among
  // WhileContexts in this graph.
  // 创建并返回一个新的WhileContext，归该计算图所有。
  // frame_name在此计算图的WhileContexts中必须是惟一的。
  Status AddWhileContext(StringPiece frame_name, std::vector<Node*> enter_nodes,
                         std::vector<Node*> exit_nodes,
                         OutputTensor cond_output,
                         std::vector<OutputTensor> body_inputs,
                         std::vector<OutputTensor> body_outputs,
                         WhileContext** result);

  // TODO(josh11b): uint64 hash() const;

 private:
  // If cost_node is non-null, then cost accounting (in CostModel)
  // will be associated with that node rather than the new one being
  // created.
  //
  // Ownership of the returned Node is not transferred to caller.
  //
  // 如果cost_node不为空，那么cost accounting(在CostModel中)将与
  // 该节点相关联，而不是与正在创建的新节点相关联。
  // 返回节点的所有权不会转移到调用方。
  Node* AllocateNode(std::shared_ptr<NodeProperties> props,
                     const Node* cost_node);
  void ReleaseNode(Node* node);

  // Registry of all known ops, including functions.
  // 注册所有已知的ops，包括functions。
  FunctionLibraryDefinition ops_;

  // GraphDef versions
  const std::unique_ptr<VersionDef> versions_;

  // Allocator which will give us good locality.
  core::Arena arena_;

  // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
  // the node with that id was removed from the graph.
  // 从节点id到节点的映射表，如果节点nodes_[id]从计算图中被删除了，那么其值有可能为空。
  std::vector<Node*> nodes_;

  // Number of nodes alive.
  // 活跃节点的数量。
  int64 num_nodes_ = 0;

  // Map from edge ids to allocated edges.  edges_[id] may be nullptr if
  // the edge with that id was removed from the graph.
  // 从edge的id到edge的映射表，如果edges_[id]从计算图中被删除了，那么其值有可能为空。
  std::vector<Edge*> edges_;

  // The number of entries in edges_ that are not nullptr.
  // 活跃edge的数量。
  int num_edges_ = 0;

  // Allocated but free nodes and edges.
  // 分配了，但后面被释放了的节点和边。
  std::vector<Node*> free_nodes_;
  std::vector<Edge*> free_edges_;

  // For generating unique names.
  // 用于生成唯一的名称。
  int name_counter_ = 0;

  // In most graphs, the number of unique values used for the
  // Node::assigned_device_name() property is quite small.  If the graph is
  // large, then this duplication of values can consume a significant amount of
  // memory.  Instead, we represent the same information using an interning
  // table, which consists of a vector of unique strings (device_names_), as
  // well a map (device_names_map_) from unique strings to indices within the
  // unique string table.
  //
  // 在大多数图中，用于Node::assigned_device_name()属性的惟一值非常少。
  // 如果计算图很大，那么这种值的重复会消耗大量内存。相反，我们使用一个外部的表
  // 去表示相同的信息，该表由惟一字符串(device_names_)组成的向量以及从惟一字符串
  // 到惟一字符串表中的索引的映射(device_names_map_)。
  //
  // The InternDeviceName() method handles adding a new entry into the table,
  // or locating the index of an existing entry.
  //
  // InternDeviceName()用于向表中添加新条目，或定位现有条目的索引。
  //
  // The fact that Node::assigned_device_name() is implemented using an
  // interning table is intentionally public.  This allows algorithms that
  // frequently access this field to do so efficiently, especially for the case
  // where the assigned_device_name of one Node is copied directly from that
  // of another Node.
  //
  // Node::assigned_device_name()是使用外部表实现的，这是有意公开的。
  // 这使得频繁访问该字段的算法能够有效地做到这一点，特别是在一个节点
  // 的assigned_device_name直接从另一个节点的assigned_device_name复制的情况下。
  //
  // A table of the unique assigned device names.  Indices do NOT correspond
  // to node IDs.  Index 0 is always the empty string.
  // 唯一指定设备名称的表。索引与节点id不相对应。索引0总是空字符串。
  std::vector<string> device_names_;

  // Maps unique device names to indices within device_names_[i].
  // 将唯一的设备名称映射到device_names_[i]中的索引。
  std::unordered_map<string, int> device_names_map_;

  // All the while contexts owned by this graph, keyed by frame name,
  // corresponding to all the while loops contained in this graph (including
  // nested loops). The stored contexts are usually accessed via
  // AddWhileContext() or Node::while_ctx(), but this manages the lifetime.
  // 所有while上下文归该计算图所拥，以帧的名字作为关键字，
  // 对应于此图所包含的所有while循环(包括嵌套循环)。
  // 存储的上下文通常通过AddWhileContext()或Node::while_ctx()访问。
  std::map<string, WhileContext> while_ctxs_;

  // Searches through edges_ for the Edge whose destination node and index
  // matches dst. An edge with destination `dst` must exist in the graph.
  // 通过edges_搜索其目标节点和索引与dst匹配的边。该边的目标dst必须存在于该计算图中。
  const Edge* FindEdge(const Node* dst, int index);

  TF_DISALLOW_COPY_AND_ASSIGN(Graph);
};

// TODO(josh11b): We may want to support keeping an index on various
// node/edge attributes in a graph, particularly node names.

// Helper routines

inline bool IsSource(const Node* node) { return node->IsSource(); }
inline bool IsSink(const Node* node) { return node->IsSink(); }
inline bool IsSwitch(const Node* node) { return node->IsSwitch(); }
inline bool IsMerge(const Node* node) { return node->IsMerge(); }
inline bool IsEnter(const Node* node) { return node->IsEnter(); }
inline bool IsExit(const Node* node) { return node->IsExit(); }
inline bool IsNextIteration(const Node* n) { return n->IsNextIteration(); }
inline bool IsLoopCond(const Node* node) { return node->IsLoopCond(); }
inline bool IsControlTrigger(const Node* n) { return n->IsControlTrigger(); }
inline bool IsSend(const Node* node) { return node->IsSend(); }
inline bool IsRecv(const Node* node) { return node->IsRecv(); }
inline bool IsHostSend(const Node* node) { return node->IsHostSend(); }
inline bool IsHostRecv(const Node* node) { return node->IsHostRecv(); }

// True for Nodes that mediate the transfer of values between processes.
inline bool IsTransferNode(const Node* n) { return IsSend(n) || IsRecv(n); }

inline bool IsConstant(const Node* node) { return node->IsConstant(); }
inline bool IsVariable(const Node* node) { return node->IsVariable(); }
inline bool IsIdentity(const Node* node) { return node->IsIdentity(); }

// Returns true iff 'n' is a control flow node.
inline bool IsControlFlow(const Node* n) { return n->IsControlFlow(); }

// Returns true if the node only depends on its input's metadata
// (shape).  Specifically, returns true for "Size", "Shape" and "Rank" ops.
inline bool IsMetadata(const Node* n) { return n->IsMetadata(); }

inline bool IsScopedAllocator(const Node* n) { return n->IsScopedAllocator(); }

inline bool IsHostMemoryPreserving(const Node* node) {
  return IsIdentity(node) || IsControlFlow(node);
}

// Iterator for stepping through the nodes of a graph.
class NodeIter {
 public:
  NodeIter(const Graph* graph, int id);
  bool operator==(const NodeIter& rhs);
  bool operator!=(const NodeIter& rhs);
  void operator++();
  Node* operator*();
  Node* operator->();

 private:
  // Invariant: id_ == graph_->num_node_ids() || graph_->FindId(id_) != nullptr
  const Graph* graph_;
  int id_;
};

// Iterator for stepping through the neighbors of a node.
class NeighborIter {
 public:
  NeighborIter(EdgeSet::const_iterator iter, bool incoming);
  bool operator==(const NeighborIter& rhs);
  bool operator!=(const NeighborIter& rhs);
  void operator++();
  Node* operator*();
  Node* operator->();

 private:
  EdgeSet::const_iterator iter_;
  bool incoming_;
};

// IMPLEMENTATION DETAILS, PLEASE IGNORE

inline NodeIter::NodeIter(const Graph* graph, int id)
    : graph_(graph), id_(id) {}

inline bool NodeIter::operator==(const NodeIter& rhs) {
  DCHECK(graph_ == rhs.graph_);
  return id_ == rhs.id_;
}

inline bool NodeIter::operator!=(const NodeIter& rhs) {
  return !(*this == rhs);
}

inline void NodeIter::operator++() {
  while (1) {
    DCHECK_LE(id_, graph_->num_node_ids());
    ++id_;
    if (id_ >= graph_->num_node_ids() || graph_->FindNodeId(id_) != nullptr) {
      return;
    }
  }
}

inline Node* NodeIter::operator*() { return graph_->FindNodeId(id_); }

inline Node* NodeIter::operator->() { return graph_->FindNodeId(id_); }

inline NeighborIter::NeighborIter(EdgeSet::const_iterator iter, bool incoming)
    : iter_(iter), incoming_(incoming) {}

inline bool NeighborIter::operator==(const NeighborIter& rhs) {
  return iter_ == rhs.iter_ && incoming_ == rhs.incoming_;
}

inline bool NeighborIter::operator!=(const NeighborIter& rhs) {
  return !(*this == rhs);
}

inline void NeighborIter::operator++() { ++iter_; }

inline Node* NeighborIter::operator*() {
  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline Node* NeighborIter::operator->() {
  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline bool Edge::IsControlEdge() const {
  // Note that if either src_output_ or dst_input_ is kControlSlot,
  // so is the other one (AddEdge checks this).
  return src_output_ == Graph::kControlSlot;
}

inline gtl::iterator_range<NodeIter> Graph::nodes() const {
  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  return gtl::make_range(NodeIter(this, 0), NodeIter(this, num_node_ids()));
}

inline gtl::iterator_range<NodeIter> Graph::op_nodes() const {
  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  //
  // The current implementation of Graph maintains the invariant that the
  // first two nodes are the source and sink nodes, and all other nodes are op
  // nodes. This method (op_nodes()) relies on this invariant.
  NodeIter begin(this, 0);
  NodeIter end(this, num_node_ids());
  if (begin != end) {
    ++begin;
  }
  if (begin != end) {
    ++begin;
  }
  return gtl::make_range(begin, end);
}

inline void Node::set_assigned_device_name_index(int index) {
  graph_->CheckDeviceNameIndex(index);
  assigned_device_name_index_ = index;
}

inline void Node::set_assigned_device_name(const string& device_name) {
  graph_->set_assigned_device_name(this, device_name);
}

inline const string& Node::assigned_device_name() const {
  return graph_->get_assigned_device_name(*this);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_H_
