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

void print_tensor(ggml_tensor *tensor, const char *name = "", int verbosity = 0)
{
    if (tensor->ne[2] == 1)
    {
        printf("---> %s: (%ld, %ld)\n", name, tensor->ne[0], tensor->ne[1]);
    }
    else if (ggml_is_3d(tensor))
    {
        printf("---> %s: (%ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
    }
    else
    {
        printf("---> %s: (%ld, %ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
    if (verbosity == 1)
    {
        printf("*********************************************************************\n");
        if (tensor->ne[2] == 1)
        {
            const float *mat = (float *)tensor->data;
            int          dim0 = tensor->ne[1];
            int          dim1 = tensor->ne[0];
            if (dim0 < 6 && dim1 < 6)
            {
                for (int i = 0; i < dim0; i++)
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        printf("%+.4f ", mat[i * dim1 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            else
            {
                for (int i = 0; i < std::min(dim0, 3); i++)
                {
                    for (int j = 0; j < std::min(dim1, 3); j++)
                    {
                        printf("%+.6f ", mat[i * dim1 + j]);
                    }
                    printf("... ");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        printf("%+.6f ", mat[i * dim1 + j]);
                    }
                    printf("\n");
                }
                if (dim0 > 3)
                {
                    printf("...................... omit ......................\n");
                    for (int i = dim0 - 3; i < dim0; i++)
                    {
                        for (int j = 0; j < std::min(dim1, 3); j++)
                        {
                            printf("%+.6f ", mat[i * dim1 + j]);
                        }
                        printf("... ");
                        for (int j = dim1 - 3; j < dim1; j++)
                        {
                            printf("%+.6f ", mat[i * dim1 + j]);
                        }
                        printf("\n");
                    }
                }
            }
        }
        else if (ggml_is_3d(tensor))
        {
            const float *data = (float *)tensor->data;
            int          dim0 = tensor->ne[2];
            int          dim1 = tensor->ne[1];
            int          dim2 = tensor->ne[0];
            if (dim0 < 6 && dim1 < 6 && dim2 < 6)
            {
                for (int i = 0; i < dim0; i++)
                {
                    printf("dim0 = %d\n", i);
                    for (int j = 0; j < dim1; j++)
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n");
            }
            else
            {
                for (int i = 0; i < std::min(dim0, 3); i++)
                {
                    printf("dim0 = %d\n", i);
                    for (int j = 0; j < std::min(dim1, 3); j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("........................\n");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("---------------------------------------------------\n");
                }
                printf("\n");
            }
        }
    }
    printf("*********************************************************************\n");
    printf("\n");
}

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

// void print_tensor(ggml_tensor * tensor, const char * name = "", int verbosity = 0){
//     if (ggml_is_3d(tensor))
//     {
//         printf("---> %s: (%ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
//     }
//     else{
//         printf("---> %s: (%ld, %ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
//     }
//     if (verbosity == 1){
//         if (ggml_is_3d(tensor))
//         {        
//             printf("------------------only show up to  3*10*10 --------------------------\n");
//             const float *data = (float *)tensor->data;
//             int          dim0 = std::min(tensor->ne[2], static_cast<int64_t>(3));
//             int          dim1 = std::min(tensor->ne[1], static_cast<int64_t>(10));
//             int          dim2 = std::min(tensor->ne[0], static_cast<int64_t>(10));
//             for (int i = 0; i < dim0; i++)
//             {
//                 printf("dim0 = %d\n", i);
//                 for (int j = 0; j < dim1; j++)
//                 {
//                     for (int k = 0; k < dim2; k++)
//                     {
//                         printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
//                     }
//                     printf("\n");
//                 }
//                 printf("\n");
//             }
//             printf("\n");
//         }
//     }
// }



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
    int dim_head=96;
    int heads=16;
    float scale = std::pow(dim_head, -0.5);
    struct ggml_tensor * mm_model_k_w;
    struct ggml_tensor * mm_model_q_w;
    struct ggml_tensor * mm_model_v_w;
    struct ggml_tensor * mm_model_o_w;
    struct ggml_tensor * mm_model_ln_media_w;
    struct ggml_tensor * mm_model_ln_media_b;
    struct ggml_tensor * mm_model_ln_latents_w;
    struct ggml_tensor * mm_model_ln_latents_b;

    // Forward
    int mult = 4;
    struct ggml_tensor * mm_model_ffn_ln_w;
    struct ggml_tensor * mm_model_ffn_ln_b;
    struct ggml_tensor * mm_model_ffn_linear_up_w;
    struct ggml_tensor * mm_model_ffn_linear_down_w;
};

struct xgenmm_perceiver_resampler {
    struct ggml_tensor * mm_model_latents;
    struct ggml_tensor * mm_model_projection_w;
    struct ggml_tensor * mm_model_projection_b;    
    std::vector<xgenmm_perceiver_resampler_layer> mm_model_layers;
    struct ggml_tensor * mm_model_norm_w;
    struct ggml_tensor * mm_model_norm_b;
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

static ggml_cgraph *build_graph(clip_ctx *ctx, ggml_tensor *img_embeddings, ggml_tensor *attn_bias_input = nullptr)
{
    const auto & model = ctx->perceiver_resampler;
    // FIXME: hard coded for now
    const float eps  = 1e-5;
    // FIXME: hard coded for now
    const int batch_size = 9;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    LOG_TEE("%s: ctx->buf_compute_meta.size(): %zu \n", __func__, ctx->buf_compute_meta.size());
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // computation starts here
    struct ggml_tensor * self_latents = model.mm_model_latents;
    

    
    
    // FIXME: hard coded for now
    int n_layer = 6;
    const float scale =  model.mm_model_layers[0].scale;
    const int num_head = model.mm_model_layers[0].heads;
    const int dim_head = model.mm_model_layers[0].dim_head;
    const int q_len = self_latents->ne[1];
    const int kv_len = img_embeddings->ne[1] + self_latents->ne[1]; // concat img_embeddings and latents
    const int hidden_size = dim_head * num_head;
    // DEBUG: remove later
    n_layer = 6;
    ggml_tensor *latents = self_latents;
    ggml_tensor *latents_repeat_along_batch = ggml_new_tensor_3d(ctx0, latents->type, latents->ne[0], latents->ne[1], batch_size);
    latents = ggml_repeat(ctx0, latents, latents_repeat_along_batch);

    ggml_tensor *ans;
    for (int il = 0; il < n_layer; ++il)
    {
        struct ggml_tensor * residual = latents;
        auto & layer = model.mm_model_layers[il];
        // layer norm

        struct ggml_tensor *img_embeddings_normalized  = ggml_norm(ctx0, img_embeddings, eps);
        img_embeddings_normalized = ggml_add(
            ctx0, ggml_mul(ctx0, img_embeddings_normalized, layer.mm_model_ln_media_w), layer.mm_model_ln_media_b);
        latents = ggml_norm(ctx0, latents, eps);
        latents = ggml_add(ctx0, ggml_mul(ctx0, latents, layer.mm_model_ln_latents_w),
                                      layer.mm_model_ln_latents_b);
        //cross attention
        {
            struct ggml_tensor *Q = ggml_mul_mat(ctx0, layer.mm_model_q_w, latents);
            Q = ggml_scale_inplace(ctx0, Q, scale);
            struct ggml_tensor *kv_inputs = ggml_concat(ctx0, img_embeddings_normalized, latents, 1);
            struct ggml_tensor * K = ggml_mul_mat(ctx0, layer.mm_model_k_w, kv_inputs);
            struct ggml_tensor * V = ggml_mul_mat(ctx0, layer.mm_model_v_w, kv_inputs);
            // permute
            Q = ggml_reshape_4d(ctx0, Q, dim_head, num_head, q_len, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, dim_head, q_len, num_head * batch_size);
            
            
            K = ggml_reshape_4d(ctx0, K, dim_head, num_head, kv_len, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, dim_head, kv_len, num_head * batch_size);
            
            V = ggml_reshape_4d(ctx0, V, dim_head, num_head, kv_len, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, kv_len, dim_head, num_head * batch_size);
        
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        
            if (attn_bias_input){
                KQ = ggml_cont(ctx0, ggml_reshape_4d(ctx0, KQ, kv_len, q_len, num_head, batch_size));
                attn_bias_input = ggml_cont(ctx0, ggml_reshape_4d(ctx0, attn_bias_input, kv_len, q_len, 1, batch_size));

                KQ = ggml_add(ctx0, KQ, attn_bias_input);

                KQ = ggml_cont(ctx0, ggml_reshape_3d(ctx0, KQ, kv_len, q_len, num_head * batch_size));
            };

            // ggml_soft_max_inplace use numerical stable softmax implementation
            // ggml_soft_max_inplace(ctx0, KQ) =  (sim - sim.amax(dim=-1, keepdim=True).detach()).softmax(dim=-1)
            KQ = ggml_soft_max_inplace(ctx0, KQ);


            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, dim_head, q_len, num_head, batch_size);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            KQV = ggml_cont_3d(ctx0, KQV, hidden_size, q_len, batch_size);


            latents = ggml_mul_mat(ctx0, layer.mm_model_o_w, KQV);
        }
        ans = latents;

        // residual connection

        latents = ggml_add(ctx0, latents, residual);
        residual = latents;  // update residual

        // FFN
        {
            // layer norm
            latents = ggml_norm(ctx0, latents, eps);
            latents = ggml_add(ctx0, ggml_mul(ctx0, latents, layer.mm_model_ffn_ln_w),
                                              layer.mm_model_ffn_ln_b);
            // feed forward
            latents = ggml_mul_mat(ctx0, layer.mm_model_ffn_linear_up_w, latents);
            latents = ggml_gelu_inplace(ctx0, latents);
            latents = ggml_mul_mat(ctx0, layer.mm_model_ffn_linear_down_w, latents);
        }
        

        // residual connection
        latents = ggml_add(ctx0, latents, residual);
    }

    // post layer norm
    latents = ggml_norm(ctx0, latents, eps);
    latents = ggml_add(ctx0, ggml_mul(ctx0, latents, model.mm_model_norm_w), model.mm_model_norm_b);
    latents = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_projection_w, latents), model.mm_model_projection_b);
    ans = latents;
    ggml_build_forward_expand(gf, ans);

    ggml_free(ctx0);
    return gf;
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
    perceiver_resampler.mm_model_latents = ggml_get_tensor(new_clip->ctx_data, "perceiver_resampler.latents");
    perceiver_resampler.mm_model_projection_w = ggml_get_tensor(new_clip->ctx_data, "perceiver_resampler.projection.weight");
    perceiver_resampler.mm_model_projection_b = ggml_get_tensor(new_clip->ctx_data, "perceiver_resampler.projection.bias");
    // FIXME: hard coded for now
    int n_layer = 6;
    perceiver_resampler.mm_model_layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il){
        auto & layer = perceiver_resampler.mm_model_layers[il];
        layer.mm_model_k_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_k", "weight"));
        layer.mm_model_q_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_q", "weight"));
        layer.mm_model_v_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_v", "weight"));
        layer.mm_model_o_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "to_out", "weight"));
        layer.mm_model_ln_media_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_media", "weight"));
        layer.mm_model_ln_media_b = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_media", "bias"));
        layer.mm_model_ln_latents_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_latents", "weight"));
        layer.mm_model_ln_latents_b = get_tensor(new_clip->ctx_data, format(TN_XGENMM_ATTN, il, "norm_latents", "bias"));
        layer.mm_model_ffn_ln_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "ln", "weight"));
        layer.mm_model_ffn_ln_b = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "ln", "bias"));
        layer.mm_model_ffn_linear_up_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "linear_up", "weight"));
        layer.mm_model_ffn_linear_down_w = get_tensor(new_clip->ctx_data, format(TN_XGENMM_FFN, il, "linear_down", "weight"));
    }
    perceiver_resampler.mm_model_norm_w = get_tensor(new_clip->ctx_data, "perceiver_resampler.ln.weight");
    perceiver_resampler.mm_model_norm_b = get_tensor(new_clip->ctx_data, "perceiver_resampler.ln.bias");

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;

    // measure mem requirement and allocate
    {
        new_clip->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        new_clip->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_clip->backend));
        // FIXME: hard coded for now
        const int ctx_size = 1152 * 729 * 10 * 9;
        struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
        };
        struct ggml_context * ctx_temp = ggml_init(params);
        // FIXME: hard coded for now
        int batch_size = 9;
        ggml_tensor *temp_image_embedding =
            ggml_new_tensor_4d(ctx_temp, GGML_TYPE_F32, 1152, 729, batch_size, 1);  // (dim, seq_len, bs)
        ggml_tensor *temp_attn_bias_input =
            ggml_new_tensor_4d(ctx_temp, GGML_TYPE_F32, 857, 128, batch_size, 1);  // (seq_len, bs)
        // ggml_tensor  *temp_vision_attn_masks = ggml_new_tensor_4d(ctx_temp, GGML_TYPE_F32, 729, 1, 1, 1); // (seq_len_image, bs)
        // ggml_tensor  *all_one_tensor = ggml_new_tensor_4d(ctx_temp, GGML_TYPE_F32, 128, 1, 1, 1); // (seq_len_query, bs)
        ggml_cgraph *gf = build_graph(new_clip, temp_image_embedding, temp_attn_bias_input);
        ggml_gallocr_reserve(new_clip->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(new_clip->compute_alloc, 0);
        LOG_TEE("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size /1024.0/1024.0);
        ggml_free(ctx_temp);
    }
    return new_clip;
}



void tensor_to_csv_3d(ggml_tensor *tensor, const char *filename)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
    }
    const float *data = (float *)tensor->data;
    int          dim0 = tensor->ne[2];
    int          dim1 = tensor->ne[1];
    int          dim2 = tensor->ne[0];
    {
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    outFile << int(data[i * dim1 * dim2 + j * dim2 + k]);
                    if (k < dim2 - 1)
                    {
                        outFile << ",";
                    }
                }
                outFile << std::endl;
            }
            outFile << std::endl;
        }
    }

    outFile.close();
    printf("file saved to %s\n", filename);
}

struct tensor_from_gguf
{
    struct ggml_tensor  *data;
    struct ggml_context *ctx;
};

bool load_tensor_from_file(const char *filename, tensor_from_gguf &tensor)
{
    struct gguf_init_params params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&tensor.ctx,
    };
    gguf_context *ctx = gguf_init_from_file(filename, params);
    if (!ctx)
    {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    tensor.data = ggml_get_tensor(tensor.ctx, "data");

    return true;
}



int main(){
    const char * fname = "/export/home/ggml/examples/projectors/projector-f32.gguf";

    struct clip_ctx * ctx = xgenmm_perceiver_resampler_load(fname);
    if (!ctx) {
        LOG_TEE("failed to load model from %s\n", fname);
        return 1;
    }
    printf("after xgenmm_perceiver_resampler_load\n");
    // int bs = 1;
    // ggml_tensor * image_embeddings = ggml_new_tensor_4d(ctx->ctx_data, GGML_TYPE_F32, 1152, 729, 1, bs);

    // std::string filename = "../examples/projectors/ggml_input.gguf";
    std::string filename = "../examples/projectors/image_embed_bs_9_cpu.gguf";
    tensor_from_gguf tensor;
    bool is_successful = load_tensor_from_file(filename.c_str(), tensor);
    if (!is_successful)
    {
        fprintf(stderr, "%s: load_tensor_from_file() failed\n", __func__);
        return 1;
    }
    ggml_tensor * image_embeddings = tensor.data;
    print_tensor(image_embeddings, "image_embeddings", 1);


    // filename = "../examples/projectors/vision_attn_masks.gguf";
    filename = "../examples/projectors/vision_attn_masks_bs_9.gguf";
    is_successful = load_tensor_from_file(filename.c_str(), tensor);
    if (!is_successful)
    {
        fprintf(stderr, "%s: load_tensor_from_file() failed\n", __func__);
        return 1;
    }
    ggml_tensor *vision_attn_masks = tensor.data;
    print_tensor(vision_attn_masks, "vision_attn_masks", 1);


    // compute attnetion masks outside of the graph
    struct ggml_tensor * attn_bias_input;
    struct ggml_context * ctx0;
    if (vision_attn_masks)
    {
        const int ctx_size = 1024 * 1024 * 1024;
        struct ggml_init_params params
        {
            /*.mem_size   =*/ctx_size,
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
        };
        ctx0 = ggml_init(params);
        // vision_attn_mask 
        // 1 -> 0
        // 0 -> -inf
        const int batch_size = vision_attn_masks->ne[1];
        const int vision_seq_length = vision_attn_masks->ne[0];
        for (int i = 0; i < batch_size * vision_seq_length; i++)
        {
            if (((float *)vision_attn_masks->data)[i] == 1.0)
            {
                ((float *)vision_attn_masks->data)[i] = 0.0;
            }
            else
            {
                ((float *)vision_attn_masks->data)[i] = -INFINITY;
            }
        }
        const int lantents_seq_length = 128;  // lantents_seq_length
        struct ggml_tensor *all_zero_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, lantents_seq_length, batch_size);
        std::fill_n((float *)all_zero_tensor->data, lantents_seq_length * batch_size, 0.0);


        vision_attn_masks = ggml_concat(ctx0, vision_attn_masks, all_zero_tensor, 0);
        print_tensor(vision_attn_masks, "vision_attn_masks append with all 1 and negate", 1);
        ggml_tensor *attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, lantents_seq_length + vision_seq_length,
                                                    batch_size, lantents_seq_length);
        print_tensor(attn_bias, "attn_bias", 0);
        attn_bias = ggml_repeat(ctx0, vision_attn_masks, attn_bias);
        attn_bias = ggml_cont(ctx0, ggml_permute(ctx0, attn_bias, 0, 2, 1, 3));

        struct ggml_cgraph *gf_temp = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf_temp, attn_bias);
        ggml_graph_compute_with_ctx(ctx0, gf_temp, 1);
        attn_bias_input = attn_bias;
    }
    print_tensor(attn_bias_input, "attn_bias_input", 1);

    // tensor_to_csv_3d(attn_bias_input, "/export/home/ggml/examples/projectors/attn_bias.csv");

    ggml_cgraph *gf = build_graph(ctx, image_embeddings, attn_bias_input);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);
    ggml_backend_graph_compute(ctx->backend, gf);

    const char *fname_cgraph = "perceiver_resampler";
    if (fname_cgraph) {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        ggml_graph_dump_dot(gf, NULL, "perceiver_resampler.dot");
        ggml_graph_export(gf, fname_cgraph);
        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }
    printf("---> number of nodes: %d\n", gf->n_nodes);
    // // before enter cross attn block; all is good
    // print_tensor(gf->leafs[0], "lantents", 1);
    // print_tensor(gf->nodes[0], "repeat lantents", 1);
    // print_tensor(gf->nodes[3], "ln lantents", 1);


    // print_tensor(gf->leafs[5], "lantents", 1);
    // print_tensor(gf->nodes[3], "repeta lantents", 1);
    // print_tensor(gf->nodes[2], "ln(x)", 1);
    // print_tensor(gf->nodes[6], "ln(latents)", 1);

    struct ggml_tensor * llm_inputs = gf->nodes[gf->n_nodes - 1];
    print_tensor(llm_inputs, "llm_inputs", 1);
    clip_free(ctx);
    ggml_free(tensor.ctx);
    if (ctx0){
        ggml_free(ctx0);
    }
    printf("Rememeber to revert ggml_vec_gelu_f32 in ggml.c after DEBUG\n");
    return 0;
}

// make perceiver_resampler -j8 && ./bin/perceiver_resampler 
// make perceiver_resampler -j8 && ./bin/perceiver_resampler && dot -Tpng perceiver_resampler.dot > perceiver_resampler.png