// #include "common.h"
// #include "log.h"
// #include "ggml.h"
// #include "ggml-alloc.h"
// #include "ggml-backend.h"
// #include "ggml-alloc.h"
#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>
// #include <stdio.h>
// #include <stdarg.h>
// #include <climits>

#define TN_XGENMM_ATTN "perceiver_resampler.blk.%d.attn.%s.%s"
#define TN_XGENMM_FFN "perceiver_resampler.blk.%d.ffn.%s.%s"
// enum projector_type {
//     PROJECTOR_TYPE_RESAMPLER,
// };

// static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
//     { PROJECTOR_TYPE_MLP, "mlp" },
//     { PROJECTOR_TYPE_LDP, "ldp" },
//     { PROJECTOR_TYPE_LDPV2, "ldpv2"},
//     { PROJECTOR_TYPE_PERCIVER_RESAMPLER, "PercevierResampler"},
// };

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}



static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }
    // printf("%s: (%ld, %ld, %ld, %ld)\n", name.c_str(), cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    return cur;
}


static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}





struct xgenmm_perceiver_resampler_layer {
    // PerceiverAttention
    int dim=1152; 
    int dim_head=16;
    int heads=96;
    int scale = std::pow(dim_head, -0.5);
    struct ggml_tensor * k_w;
    struct ggml_tensor * q_w;
    struct ggml_tensor * v_w;
    struct ggml_tensor * o_w;
    struct ggml_tensor * ln_meida_w;
    struct ggml_tensor * ln_meida_b;
    struct ggml_tensor * ln_latents_w;
    struct ggml_tensor * ln_latents_b;

    // Forward
    int mult = 4;
    struct ggml_tensor * ffn_ln_w;
    struct ggml_tensor * ffn_ln_b;
    struct ggml_tensor * ffn_linear_up_w;
    struct ggml_tensor * ffn_linear_down_w;
};

struct xgenmm_perceiver_resampler {
    struct ggml_tensor * latents;
    struct ggml_tensor * projection_w;
    struct ggml_tensor * projection_b;    
    std::vector<xgenmm_perceiver_resampler_layer> layers;
    struct ggml_tensor * norm_w;
    struct ggml_tensor * norm_b;
};

struct clip_ctx {
    struct xgenmm_perceiver_resampler perceiver_resampler;
    bool use_gelu = false;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;
    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer  = NULL;
    ggml_backend_t backend       = NULL;
    ggml_gallocr_t compute_alloc = NULL;
};

void clip_free(clip_ctx * ctx) {
    ggml_free(ctx->ctx_data);
    gguf_free(ctx->ctx_gguf);

    ggml_backend_buffer_free(ctx->params_buffer);
    ggml_backend_free(ctx->backend);
    ggml_gallocr_free(ctx->compute_alloc);
    delete ctx;
}

// read and create ggml_context containing the tensors and their data
struct clip_ctx * xgenmm_perceiver_resampler_load(const char * fname, const int verbosity = 1){
    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname, params);
    if (!ctx) {
        throw std::runtime_error(format("%s: failed to load xgenmm_perceiver_resampler_load model from %s. Does this file exist?\n", __func__, fname));
    }

    const int n_tensors = gguf_get_n_tensors(ctx);

    // kv
    const int n_kv = gguf_get_n_kv(ctx);
    LOG_TEE("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n",
        __func__, n_kv, n_tensors, fname);
    {
        std::map<enum ggml_type, uint32_t> n_type;

        for (int i = 0; i < n_tensors; i++) {
            enum ggml_type type = gguf_get_tensor_type(ctx, i);

            n_type[type]++;
        }

        LOG_TEE("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(ctx, i);
            const enum gguf_type type   = gguf_get_kv_type(ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(ctx, i)), gguf_get_arr_n(ctx, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(ctx, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LOG_TEE("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LOG_TEE("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    // data
    size_t model_size = 0;
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);
            enum ggml_type type = gguf_get_tensor_type(ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(meta, name);
            size_t tensor_size = ggml_nbytes(cur);
            model_size += tensor_size;
            if (verbosity >= 3) {
                LOG_TEE("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                       __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    clip_ctx * new_clip = new clip_ctx;

    if (!new_clip->backend) {
        new_clip->backend = ggml_backend_cpu_init();
        LOG_TEE("%s: CLIP using CPU backend\n", __func__);
    }
    


    LOG_TEE("%s: params backend buffer size = % 6.2f MB (%i tensors)\n", __func__, model_size / (1024.0 * 1024.0), n_tensors);

     // load tensors
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            /*.mem_size =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        new_clip->ctx_data = ggml_init(params);
        if (!new_clip->ctx_data) {
            LOG_TEE("%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            gguf_free(ctx);
            return nullptr;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            LOG_TEE("cannot open model file for loading tensors\n");
            clip_free(new_clip);
            gguf_free(ctx);
            return nullptr;
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_clip->ctx_data, t);
            ggml_set_name(cur, name);
        }

        // alloc memory and offload data
        new_clip->params_buffer = ggml_backend_alloc_ctx_tensors(new_clip->ctx_data, new_clip->backend);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            // printf("name: %s\n", name);
            struct ggml_tensor * cur = ggml_get_tensor(new_clip->ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                LOG_TEE("%s: failed to seek for tensor %s\n", __func__, name);
                clip_free(new_clip);
                gguf_free(ctx);
                return nullptr;
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(new_clip->params_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();
    }

    //load tensors into the model
    auto & perceiver_resampler = new_clip->perceiver_resampler;
    perceiver_resampler.latents = ggml_get_tensor(new_clip->ctx_data, "perceiver_resampler.latents");
    perceiver_resampler.projection_w = ggml_get_tensor(new_clip->ctx_data, "perceiver_resampler.projection.weight");
    perceiver_resampler.projection_b = ggml_get_tensor(new_clip->ctx_data, "perceiver_resampler.projection.bias");
    // FIXME: hard coded for now
    int n_layer = 6;
    perceiver_resampler.layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il){
        auto & layer = perceiver_resampler.layers[il];
        layer.k_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_k", "weight"));
        layer.q_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_q", "weight"));
        layer.v_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_v", "weight"));
        layer.o_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_out", "weight"));
        layer.ln_meida_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_media", "weight"));
        layer.ln_meida_b = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_media", "bias"));
        layer.ln_latents_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_latents", "weight"));
        layer.ln_latents_b = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_latents", "bias"));
        layer.ffn_ln_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "ln", "weight"));
        layer.ffn_ln_b = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "ln", "bias"));
        layer.ffn_linear_up_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "linear_up", "weight"));
        layer.ffn_linear_down_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "linear_down", "weight"));
    }
    perceiver_resampler.norm_w = get_tensor(new_clip->ctx_data, "perceiver_resampler.ln.weight");
    perceiver_resampler.norm_b = get_tensor(new_clip->ctx_data, "perceiver_resampler.ln.bias");

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;
    new_clip->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    return new_clip;
}


static ggml_cgraph * build_graph(clip_ctx * ctx, ggml_tensor * embeddings){
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    LOG_TEE("%s: ctx->buf_compute_meta.size(): %zu \n", __func__, ctx->buf_compute_meta.size());
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);



    ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);
    return gf;
}




int main(){
    const char * fname = "/export/home/ggml/examples/projectors/projector-f32.gguf";
    struct clip_ctx * ctx = xgenmm_perceiver_resampler_load(fname);
    if (!ctx) {
        LOG_TEE("failed to load model from %s\n", fname);
        return 1;
    }
    ggml_tensor * embeddings = ggml_new_tensor_2d(ctx->ctx_data, GGML_TYPE_F32, 1, 1);
    ggml_cgraph * gf = build_graph(ctx, embeddings);
    return 0;
}


// make perceiver_resampler -j8 && ./bin/perceiver_resampler