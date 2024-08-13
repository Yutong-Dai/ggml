#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "ggml.h"

void print_tensor(ggml_tensor* tensor, const char* name = "", int verbosity = 0)
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
            const float* mat = (float*)tensor->data;
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
                        printf("%+.4f ", mat[i * dim1 + j]);
                    }
                    printf("... ");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        printf("%+.4f ", mat[i * dim1 + j]);
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
                            printf("%+.4f ", mat[i * dim1 + j]);
                        }
                        printf("... ");
                        for (int j = dim1 - 3; j < dim1; j++)
                        {
                            printf("%+.4f ", mat[i * dim1 + j]);
                        }
                        printf("\n");
                    }
                }
            }
        }
        else if (ggml_is_3d(tensor))
        {
            const float* data = (float*)tensor->data;
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
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
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
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("........................\n");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
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

struct tensor_from_gguf
{
    struct ggml_tensor*  data;
    struct ggml_context* ctx;
};

bool load_tensor_from_file(const char* filename, tensor_from_gguf &tensor)
{
    struct gguf_init_params params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&tensor.ctx,
    };
    gguf_context*       ctx = gguf_init_from_file(filename, params);
    if (!ctx)
    {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    tensor.data = ggml_get_tensor(tensor.ctx, "data");

    return true;
}

struct data_struct
{
    struct ggml_tensor*  data;
    struct ggml_tensor*  w;
    struct ggml_tensor* ln_w;
    struct ggml_tensor* ln_b;
    struct ggml_context* ctx;
};

bool load_data_struct_from_file(const char* filename, data_struct &model)
{
    struct gguf_init_params params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&model.ctx,
    };
    gguf_context*       ctx = gguf_init_from_file(filename, params);
    if (!ctx)
    {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.data = ggml_get_tensor(model.ctx, "data");
    model.w = ggml_get_tensor(model.ctx, "w");
    model.ln_w = ggml_get_tensor(model.ctx, "ln_w");
    model.ln_b = ggml_get_tensor(model.ctx, "ln_b");

    return true;
}

int main()
{
    std::string filename = "../examples/projectors/ggml_input.gguf";
    // tensor_from_gguf tensor;
    // bool is_successful = load_tensor_from_file(filename.c_str(), tensor);
    // if (!is_successful)
    // {
    //     fprintf(stderr, "%s: load_tensor_from_file() failed\n", __func__);
    //     return 1;
    // }
    // print_tensor(tensor.data, "tensor", 1);
    // ggml_free(tensor.ctx);

    data_struct model;
    bool is_successful = load_data_struct_from_file(filename.c_str(), model);
    if (!is_successful)
    {
        fprintf(stderr, "%s: load_data_struct_from_file() failed\n", __func__);
        return 1;
    }
    print_tensor(model.data, "data", 1);
    print_tensor(model.w, "w", 1);
    print_tensor(model.ln_w, "ln_w", 1);
    print_tensor(model.ln_b, "ln_b", 1);

    struct ggml_init_params params
    {
        /*.mem_size   =*/1024 * 1024 * 1024,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };
    struct ggml_context* ctx0 = ggml_init(params);
    
    printf("-----------------------------------------------\n");
    struct ggml_tensor* Q = ggml_norm(ctx0, model.data, 1e-5);
    Q = ggml_add(ctx0, ggml_mul(ctx0, Q, model.ln_w), model.ln_b);
    struct ggml_tensor* P = ggml_mul_mat(ctx0, model.w, Q);

    struct ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, P);
    ggml_graph_compute_with_ctx(ctx0, gf, 1);


    const char *fname_cgraph = "io";
    if (fname_cgraph) {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        ggml_graph_dump_dot(gf, NULL, "io.dot");
        ggml_graph_export(gf, fname_cgraph);
        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

    print_tensor(gf->leafs[1], "data", 1);
    // print_tensor(gf->leafs[2], "ln_b", 1);
    print_tensor(gf->nodes[0], "After norm(mean, var)", 1);
    print_tensor(gf->nodes[2], "After layer_norm", 1);

    ggml_free(model.ctx);
    ggml_free(ctx0);
}

// make io -j8 && ./bin/io
// dot -Tpng io.dot > io.png