#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>

void print_mat(ggml_tensor* tensor, const char* name = "")
{
    const float* mat = (float*)tensor->data;
    int          ncols = tensor->ne[0];
    int          nrows = tensor->ne[1];
    printf("%s: ne[0]=%d, ne[1]=%d\n------------\n", name, ncols, nrows);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            printf("%.2f ", mat[i * ncols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_tensor_3d(ggml_tensor* tensor, const char* name = "")
{
    const float* data = (float*)tensor->data;
    int          dim0 = tensor->ne[2];
    int          dim1 = tensor->ne[1];
    int          dim2 = tensor->ne[0];
    printf("%s: ne[0]=%ld, ne[1]=%ld, ne[2]=%ld, ne[3]=%ld\n---------------------------------------------\n", name,
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    for (int i = 0; i < dim0; i++)
    {
        printf("dim0 = %d\n", i);
        for (int j = 0; j < dim1; j++)
        {
            for (int k = 0; k < dim2; k++)
            {
                printf("%.2f ", data[i * dim1 * dim2 + j * dim2 + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

int main(){
    const int ctx_size = 1024 * 1024;
    struct ggml_init_params params
    {
        /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };
    struct ggml_context* ctx = ggml_init(params);

    float A_data[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    struct ggml_tensor * A = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4);
    memcpy(A->data, A_data, 24 * ggml_type_size(GGML_TYPE_F32));
    print_tensor_3d(A, "A");
    int dim_head = 2;
    int num_head = 3;
    int bs = 4;

    struct ggml_tensor* A_reshaped = ggml_reshape_3d(ctx, A, dim_head, num_head, bs);
    print_tensor_3d(A_reshaped, "A_reshaped");
    
    struct ggml_tensor* A_permute = ggml_permute(ctx, A_reshaped, 0, 2, 1, 3); // A_permute (2, 4, 3)
    print_tensor_3d(A_permute, "A_permute");
    struct ggml_tensor* A_permute_cont = ggml_cont(ctx, A_permute);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* mat_2d = ggml_reshape_2d(ctx, A_permute_cont, dim_head, num_head * bs);

    ggml_build_forward_expand(gf, mat_2d);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    print_tensor_3d(A_permute_cont, "A_permute_cont");
    print_mat(mat_2d, "mat_2d");

    // const char *fname_cgraph = "shape_ops";
    // if (fname_cgraph) {
    //     // export the compute graph for later use
    //     // see the "mnist-cpu" example
    //     ggml_graph_dump_dot(gf, NULL, "shape_ops.dot");
    //     ggml_graph_export(gf, fname_cgraph);
    //     fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    // }

    ggml_free(ctx);
    return 0;


    // A = ggml_reshape_3d(ctx, A, dim_head, num_head, bs);  // A_reshaped (2, 3, 4, 1)
    // print_tensor_3d(A, "A_reshaped");
    // A = ggml_cont(ctx, ggml_permute(ctx, A, 0, 2, 1, 3));  // A_permute (2, 4, 3, 1)
    // A = ggml_reshape_2d(ctx, A, dim_head, num_head * bs);  // mat_2d (2, 12, 1, 1)
    // struct ggml_cgraph* gf = ggml_new_graph(ctx);
    // ggml_build_forward_expand(gf, A);
    // ggml_graph_compute_with_ctx(ctx, gf, 1);
    // print_mat(A, "A_2d");

    // const char *fname_cgraph = "shape_ops";
    // if (fname_cgraph) {
    //     // export the compute graph for later use
    //     // see the "mnist-cpu" example
    //     ggml_graph_dump_dot(gf, NULL, "shape_ops.dot");
    //     ggml_graph_export(gf, fname_cgraph);
    //     fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    // }

    // ggml_free(ctx);


    // return 0;

    
}

// make shape_ops -j8 && ./bin/shape_ops
// make shape_ops -j8 && ./bin/shape_ops && dot -Tpng shape_ops.dot > shape_ops.png