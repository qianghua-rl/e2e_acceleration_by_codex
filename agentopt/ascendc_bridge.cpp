#include <torch/extension.h>
#include <acl/acl.h>
#include "aclnn_gdn_conv_state_update.h"
#include <map>
#include <mutex>
#include <sstream>

static aclDataType acl_dtype(torch::ScalarType t) {
    if (t == torch::kFloat16) return ACL_FLOAT16;
    if (t == torch::kBFloat16) return ACL_BF16;
    if (t == torch::kFloat) return ACL_FLOAT;
    throw std::runtime_error("unsupported dtype");
}

static aclTensor *tensor(torch::Tensor t) {
    auto sizes = t.sizes().vec();
    auto strides = t.strides().vec();
    return aclCreateTensor(sizes.data(), sizes.size(), acl_dtype(t.scalar_type()), strides.data(), 0,
                           ACL_FORMAT_ND, sizes.data(), sizes.size(), t.data_ptr());
}

struct CachedOp {
    aclTensor *x, *scale, *weight, *state, *y;
    aclOpExecutor *executor;
    void *workspace;
    uint64_t workspace_size;
};

static std::mutex cache_mutex;
static std::map<std::string, CachedOp> cache;

static std::string cache_key(const torch::Tensor& x, const torch::Tensor& scale,
                             const torch::Tensor& weight, const torch::Tensor& state) {
    std::ostringstream os;
    os << x.scalar_type() << ':';
    for (auto v : x.sizes()) os << v << ',';
    os << '|'; for (auto v : scale.sizes()) os << v << ',';
    os << '|'; for (auto v : weight.sizes()) os << v << ',';
    os << '|'; for (auto v : state.sizes()) os << v << ',';
    return os.str();
}

torch::Tensor gdn_conv_state(torch::Tensor x, torch::Tensor scale, torch::Tensor weight, torch::Tensor state, uint64_t stream_ptr) {
    TORCH_CHECK(x.device().type() == torch::kPrivateUse1, "NPU tensor required");
    TORCH_CHECK(x.is_contiguous() && scale.is_contiguous() && weight.is_contiguous() && state.is_contiguous());
    auto y = torch::empty_like(x);
    const auto key = cache_key(x, scale, weight, state);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    aclnnStatus status;
    if (it == cache.end()) {
        CachedOp op{tensor(x), tensor(scale), tensor(weight), tensor(state), tensor(y), nullptr, nullptr, 0};
        status = aclnnGdnConvStateUpdateGetWorkspaceSize(op.x, op.scale, op.weight, op.state, op.y,
                                                          &op.workspace_size, &op.executor);
        TORCH_CHECK(status == ACL_SUCCESS, "GetWorkspaceSize failed: ", status);
        if (op.workspace_size) TORCH_CHECK(aclrtMalloc(&op.workspace, op.workspace_size,
                                                        ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        status = aclSetAclOpExecutorRepeatable(op.executor);
        TORCH_CHECK(status == ACL_SUCCESS, "executor repeatable setup failed: ", status);
        it = cache.emplace(key, op).first;
    } else {
        auto &op = it->second;
        TORCH_CHECK(aclSetInputTensorAddr(op.executor, 0, op.x, x.data_ptr()) == ACL_SUCCESS);
        TORCH_CHECK(aclSetInputTensorAddr(op.executor, 1, op.scale, scale.data_ptr()) == ACL_SUCCESS);
        TORCH_CHECK(aclSetInputTensorAddr(op.executor, 2, op.weight, weight.data_ptr()) == ACL_SUCCESS);
        TORCH_CHECK(aclSetInputTensorAddr(op.executor, 3, op.state, state.data_ptr()) == ACL_SUCCESS);
        TORCH_CHECK(aclSetOutputTensorAddr(op.executor, 0, op.y, y.data_ptr()) == ACL_SUCCESS);
    }
    auto &op = it->second;
    status = aclnnGdnConvStateUpdate(op.workspace, op.workspace_size, op.executor,
                                      reinterpret_cast<aclrtStream>(stream_ptr));
    TORCH_CHECK(status == ACL_SUCCESS, "launch failed: ", status);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_conv_state", &gdn_conv_state); }
