#include "ggml.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>


struct mul_mat_model {
    struct ggml_tensor * A;
    struct ggml_tensor * B;
    struct ggml_context * ctx;
};



bool mul_mat_model_load(const std::string & fname, mul_mat_model & model) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.A = ggml_get_tensor(model.ctx, "A");
    model.B = ggml_get_tensor(model.ctx, "B");
    return true;
}

int main(int argc, char ** argv) {

    mul_mat_model model;

    {
        if (!mul_mat_model_load(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, argv[1]);
            return 1;
        }
    }
    
    int rows_A = model.A->ne[1];
    int cols_A = model.A->ne[0];
    fprintf(stdout, "rows_A = %d, cols_A = %d\n", rows_A, cols_A);
    printf("A matrix:\n");
    float * tensor_data_A = (float *) model.A->data;
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_A; j++) {
            printf("%.2f ", tensor_data_A[i * cols_A + j]);
        }
        printf("\n");
    }

    int rows_B = model.B->ne[1];
    int cols_B = model.B->ne[0];
    fprintf(stdout, "rows_B = %d, cols_B = %d\n", rows_B, cols_B);
    printf("B matrix:\n");
    float * tensor_data_B = (float *) model.B->data;
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; j++) {
            printf("%.2f ", tensor_data_B[i * cols_B + j]);
        }
        printf("\n");
    }

    struct ggml_init_params params = { 1024 * 1024 }; // Adjust the memory size as needed
    struct ggml_context * ctx0 = ggml_init(params);
    ggml_tensor* BT = ggml_transpose(ctx0, model.B);
    int rows_BT = BT->ne[1];
    int cols_BT = BT->ne[0];
    fprintf(stdout, "rows_BT = %d, cols_BT = %d\n", rows_BT, cols_BT);
    printf("B.T matrix:\n");
    float * tensor_data_BT = (float *) BT->data;
    for (int i = 0; i < rows_BT; i++) {
        for (int j = 0; j < cols_BT; j++) {
            printf("%.2f ", tensor_data_BT[i * cols_BT + j]);
        }
        printf("\n");
    }


//     // Perform matrix multiplication
//     struct ggml_tensor * C = ggml_mul_mat(ctx0, model.A, BT);
//     ggml_set_name(C, "C");

//     struct ggml_cgraph * gf = ggml_new_graph(ctx0);
//     ggml_build_forward_expand(gf, C);
//     int n_threads = 1;
//     ggml_graph_compute_with_ctx(ctx0, gf, n_threads);


//    // Print the result
//     float * result = (float *) C->data;
//     printf("Result of A * B:\n");
//     for (int i = 0; i < rows_A; i++) {
//         for (int j = 0; j < cols_B; j++) {
//             printf("%f ", result[i * cols_B + j]);
//         }
//         printf("\n");
//     }


    ggml_free(model.ctx);
    ggml_free(ctx0);
}


// make mul_mat_from_gguf -j8 && ./bin/mul_mat_from_gguf /export/home/ggml/examples/tensor_operations/data_gguf/mul_mat_data.gguf