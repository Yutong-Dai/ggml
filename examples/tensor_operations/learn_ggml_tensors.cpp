#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

void print_mat(ggml_tensor * tensor, const char * name = ""){
    const float * mat = (float *) tensor->data;
    int ncols = tensor->ne[0]; 
    int nrows = tensor->ne[1];
    printf("%s: ne[0]=%d, ne[1]=%d\n------------\n", name, ncols, nrows);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            printf("%.2f ", mat[i * ncols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_tensor_3d(ggml_tensor * tensor, const char * name = ""){
    const float * data = (float *) tensor->data;
    int dim0 = tensor->ne[2];
    int dim1 = tensor->ne[1];
    int dim2 = tensor->ne[0];
    printf("%s: ne[0]=%ld, ne[1]=%ld, ne[2]=%ld, ne[3]=%ld\n---------------------------------------------\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    
    for (int i = 0; i < dim0; i++) {
        printf("dim0 = %d\n", i);
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                printf("%.2f ", data[i * dim1 * dim2 + j * dim2 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
}


struct tensor_3d_from_gguf {
    struct ggml_tensor * data;
    struct ggml_context * ctx;
};

struct bmm_model {
    struct ggml_tensor * U;
    struct ggml_tensor * V;
    struct ggml_context * ctx;
};



int main(){
    const int ctx_size = 1024 * 1024;
    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };
    struct ggml_context * ctx = ggml_init(params);
    /* 
    ggml tensors support up to 4 dimensions; 
    A Pytorch matrix of shape [m, n] is equivalent to the ggml tensors [n, m]
    A Pytorch tensors of shape [dim0, dim1, dim2, dim3] is equivalent to the ggml tensors [ne3, ne2, ne1, ne0]
    */
    const int ncols = 2;
    const int nrows = 4;
    ggml_tensor * Amat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    printf("Syntax ggml_new_tensor_2d(ctx, GGML_TYPE_F32, %d, %d) gives\n", ncols, nrows);
    printf("Amat->ne[0] = %ld, Amat->ne[1] = %ld, Amat->ne[2] = %ld, Amat->ne[3] = %ld \n", Amat->ne[0], Amat->ne[1], Amat->ne[2], Amat->ne[3]);
    /* 1 fp32 = 4 bytes*/
    size_t tensor_usage = ncols * nrows * ggml_type_size(GGML_TYPE_F32);
    printf("Memory used in bytes:\n");
    printf("    tensor_usage = %zu\n", tensor_usage);
    size_t tensor_overhead = ggml_tensor_overhead();
    printf("    tensor_overhead = %zu\n", tensor_overhead);
    size_t total_used = ggml_used_mem(ctx);
    printf("  total_used = %zu\n", total_used);


    printf("Matrix (2-D tensor experiments):\n");
    float A_data[ncols * nrows] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6};

    memcpy(Amat->data, A_data, ncols * nrows * ggml_type_size(GGML_TYPE_F32));
    print_mat(Amat, "A matrix");

    /* some basic operations on tensors*/
    /*ggml_transpose*/
    ggml_tensor * Amat_T = ggml_transpose(ctx, Amat);
    print_mat(Amat_T, "A matrix transposed");

    /*ggml_reshape_2d*/
    ggml_tensor * Amat_reshaped = ggml_reshape_2d(ctx, Amat, nrows, ncols);
    print_mat(Amat_reshaped, "A matrix reshaped");

    /*ggml_mul_mat*/
    ggml_tensor * xvec = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, 1);
    float x_data[nrows] = {1, 2};
    memcpy(xvec->data, x_data, nrows * ggml_type_size(GGML_TYPE_F32));
    print_mat(xvec, "x vector");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
      
    ggml_tensor * yvec = ggml_mul_mat(ctx, Amat, xvec);
    ggml_build_forward_expand(gf, yvec);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    print_mat(yvec, "y vector");

    const int rows_B = 3, cols_B = 2;
    ggml_tensor * Bmat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
    float B_data[cols_B * rows_B] = {
        10, 5,
        9, 9,
        5, 4};
    memcpy(Bmat->data, B_data, cols_B * rows_B * ggml_type_size(GGML_TYPE_F32));
    print_mat(Bmat, "B matrix");
    ggml_tensor * C = ggml_mul_mat(ctx, Amat, Bmat);
    ggml_build_forward_expand(gf, C);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    print_mat(C, "C matrix = A@B");

    printf("Tensor level experiments:\n");
    std::string fname = "../examples/tensor_operations/data_gguf/3d_tensor.gguf";
    tensor_3d_from_gguf tensor_3d;
    struct gguf_init_params params_new = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tensor_3d.ctx,
    };

    gguf_context * ctx_0 = gguf_init_from_file(fname.c_str(), params_new);
    if (!ctx_0) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return 1;
    }
    tensor_3d.data = ggml_get_tensor(tensor_3d.ctx, "data");
    ggml_tensor * D = tensor_3d.data;
    print_tensor_3d(D, "D(3d_tensor)");
    ggml_tensor * D_permute = ggml_permute(ctx, D, 1, 0, 2, 3);
    print_tensor_3d(D_permute, "D_permute(3d_tensor)");
    ggml_free(tensor_3d.ctx);

    /*bmm*/
    bmm_model model;
    fname = "../examples/tensor_operations/data_gguf/batched_mat_mul.gguf";
    params_new = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx_1 = gguf_init_from_file(fname.c_str(), params_new);
    if (!ctx_1) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return 1;
    }
    model.U = ggml_get_tensor(model.ctx, "U");
    model.V = ggml_get_tensor(model.ctx, "V");
    print_tensor_3d(model.U, "U(3d_tensor)");
    print_tensor_3d(model.V, "V(3d_tensor)");
    ggml_tensor * W = ggml_mul_mat(ctx, model.U, model.V);
    ggml_build_forward_expand(gf, W);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    print_tensor_3d(W, "W tensor = torch.bmm(U, V)");
    
    ggml_free(model.ctx);



    
    ggml_free(ctx);
    return 0;
}

// make learn_ggml_tensors -j8 && ./bin/learn_ggml_tensors