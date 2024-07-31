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

/*print buf_size in MB*/
void print_buf_size_in_MB(size_t buf_size){
    printf("Allocating buf_size = %f MB\n", buf_size * 1.0 / 1024 / 1024);
}


int main() {
    ggml_time_init();

    printf("ggml_mul_mat\n");
    /*
     ggml_mul_mat performs batched matrix multiplication
     struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) 
    a and b are up to 4D tensors
    */

    /* memory allocation of tensors A and B of dtype fp32*/
    const int Arow = 4;
    const int Acol = 2;
    float A_data[Arow * Acol] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6};
    const int Brow = 3;
    const int Bcol = 2;
    float B_data[Brow * Bcol] = {
        10, 5,
        9, 9,
        5, 4
    };
    
    printf("ggml_type_size(GGML_TYPE_F32): %zu\n", ggml_type_size(GGML_TYPE_F32));
    size_t ctx_size = 0;
    {
        ctx_size += Arow * Acol * ggml_type_size(GGML_TYPE_F32); // tensor a
        ctx_size += Brow * Bcol * ggml_type_size(GGML_TYPE_F32); // tensor b
        ctx_size += 2 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
    }
    static void * buf = malloc(ctx_size);
    print_buf_size_in_MB(ctx_size);

 
    struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ buf,
            /*.no_alloc   =*/ false,
        };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_tensor * A = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Acol, Arow);
    ggml_tensor * B = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Bcol, Brow);

    memcpy(A->data, A_data, ggml_nbytes(A));
    memcpy(B->data, B_data, ggml_nbytes(B));

    // build the compute graph to perform a matrix multiplication
    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);
    struct ggml_tensor * result_ = ggml_mul_mat(ctx0, A, B);
    ggml_build_forward_expand(gf, result_);
    ggml_graph_compute_with_ctx(
        ctx0, 
        gf,
        1);
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];

    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");



    ggml_free(ctx0);
    return 0;
}