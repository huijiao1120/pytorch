#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mkldnn/Utils.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/library.h>

#if AT_ONEDNN_GRAPH_ENABLED()

namespace std {
template <>
struct hash<std::vector<int64_t>> {
  size_t operator()(const std::vector<int64_t>& key) const {
    size_t total = key.size();
    size_t sum = 0;
    if (total < 64) {
      for (size_t i = 0; i < total; i++) {
        sum += key[i] << i;
      }
    } else {
      size_t batch = total / 64;
      size_t remaining = total % 64;
      for (size_t bs = 0; bs < batch; bs++) {
        for (size_t i = 0; i < 64; i++) {
          sum += key[bs * 64 + i] << i;
        }
      }
      for (size_t i = 0; i < remaining; i++) {
        sum += key[batch * 64 + i] << i;
      }
    }
    return sum;
  }
};

} // namespace std

namespace at {
namespace native {
namespace onednn_graph {

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using LogicalTensors = std::vector<logical_tensor>;

// Compiled Partition entry
struct cp_entry {
  partition partition_;
  compiled_partition cp_;
  RunArgs constantInputTensors_;
  RunArgs inputLLGATensors_;
  RunArgs outputLLGATensors_;
  LogicalTensors inputLogicalTensors_;
  LogicalTensors outputLogicalTensors_;
};

using key_value_pair_t = std::pair<std::vector<int64_t>, cp_entry>;
using list_iterator_t = std::list<key_value_pair_t>::iterator;

// define all patterns here
#define ONEDNN_GRAPH_SDPA_PATTERN_5_FP32 0
// Due to a bug in Graph Compiler, SDPA pattern 5 doesn't work
// but it'd be supported in oneDNN v3.5
#define ONEDNN_GRAPH_SDPA_PATTERN_5_BF16 1

// Somehow, linking fails if extern is not used here, although
// extern should have been implicit.

extern void insert_in_fused_kernel_cache(
    std::vector<int64_t>& map_key,
    cp_entry& cp);

extern void insert_in_partition_cache(int64_t partitionID, partition& p);

extern void change_pos_in_list(list_iterator_t& kvpair);

extern std::unordered_map<int64_t, dnnl::graph::partition>::iterator
partition_map_lookup(int64_t patternID);

extern std::unordered_map<int64_t, dnnl::graph::partition>::iterator
partition_map_end();

extern std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator
cache_lookup(std::vector<int64_t>& map_key);

extern std::unordered_map<std::vector<int64_t>, list_iterator_t>::iterator
cache_end();

extern compiled_partition compile_partition(
    const partition& partition,
    const std::vector<logical_tensor>& inputs,
    const std::vector<logical_tensor>& outputs);

} // end namespace onednn_graph
} // end namespace native
} // end namespace at

#endif
