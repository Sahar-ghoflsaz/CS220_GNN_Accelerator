// some pre-defined libraries
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>

//including some self-defined functions

#include "../support/common.h"
#include "../support/matrix.h"
#include "../support/params.h"
#include "../support/partition.h"
#include "../support/timer.h"
#include "../support/utils.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/spmv_dpu"
#endif

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MB

/*
First input: Sparse Matrix of size M * M
Second input: Dene Matrix of size M * K
Output: Matrix size of M * K

*/

static struct COOMatrix* sparseA;
static int8_t** denseB;
static int8_t** denseO;
#define FEATURE 6
// static struct partitio_info_t *part_info;

struct dpu_info_type{
    uint32_t cols_per_dpu;
    uint32_t cols_per_dpu_pad;
    uint32_t prev_cols_dpu

};
struct dpu_info_type *dpu_info;

void init_denseMatrix(int8_t** denseB, int row, int col){
    for(unsigned int i = 0; i<col; i++){
        for(unsigned int j=0; j<row; j++){
            denseB[i][j] = (int8_t) (i%4+1);
        }
    }
}

int main(int argc, char **argv){

    struct Params p = input_params(argc,argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    
    sparseA = readCOOMatrix(p.fileName);
    // sortCOOMatrix(sparseA);
    unsigned int m = sparseA->nrows;
    unsigned int f =  FEATURE;
    unsigned int nnz =  sparseA->nnz;

    printf("m:%d , f:%d, nnz:%d\n",m,f,nnz);
    int alloc_dpu = 0;
    //allocating dpus aand loading binary
    if(f < 2048) alloc_dpu = f;
    else alloc_dpu = 2048; 
    DPU_ASSERT(dpu_alloc(alloc_dpu, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("[INFO] Allocated %d DPU(s)\n", nr_of_dpus);
    printf("[INFO] Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);

    
    //this matrix iss coming from the python code and the structure is like row, col, val
    
    // part_info =  partition_init(nr_of_dpus,NR_TASKLETS);
    // partition_by_row(sparseA, part_info, nr_of_dpus);
    // printf("1\n");
    //allocate matrix b
    denseB = (int8_t**) malloc(f*sizeof(int8_t*));
    for( unsigned int i = 0; i<f; i++ ){
        denseB[i] = (int8_t*) malloc(m*sizeof(int8_t));
    }

    denseO = (int8_t**) malloc(f*sizeof(int8_t*));
    for( unsigned int i = 0; i<f; i++ ){
        denseO[i] = (int8_t*) malloc(m*sizeof(int8_t));
    }
    // printf("2\n");

    // init_denseMatrix(denseO, m, f);
    //initialize matrix b
    init_denseMatrix(denseB, m, f);
    // for(int i=0; i<m; i++){
    //     for(int j=0; j<f;j++){
    //         printf(" %d ", denseB[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for(int i=0; i<nnz; i++){

    //     printf("row:%d, col:%d, val:%d ", sparseA->nnzs[i].rowind, sparseA->nnzs[i].colind,sparseA->nnzs[i].val);
    
    // }
    // printf("\n");
    //initiallize some information needed for partitioning
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_type)); 
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    // Max limits for parallel transfers
    uint64_t max_cols_per_dpu= 0;
    uint64_t max_cols_per_tasklet = 0;
    // printf("3\n");
    Timer timer;

    unsigned int i = 0;
    DPU_FOREACH(dpu_set,dpu,i){

        uint32_t cols_per_dpu= 1;//part_info->row_split[i+1] - part_info->row_split[i];
        uint32_t prev_cols_dpu = 1;//part_info->row_split[i];
        
        // if (cols_per_dpu> max_cols_per_dpu)
        //     max_cols_per_dpu= cols_per_dpu;
        
        dpu_info[i].cols_per_dpu= cols_per_dpu;
        dpu_info[i].prev_cols_dpu = prev_cols_dpu;

        input_args[i].nrows = m;
        input_args[i].tcols = 1; 
        input_args[i].nnz = sparseA->nnz;
        // input_args[i].tstart_col = dpu_info[i].prev_cols_dpu;

    }
    // printf("4\n");
    // if (max_cols_per_dpu % 2 == 1) 
    //     max_cols_per_dpu++;
    // if (max_nnz_per_dpu % (8 / byte_dt) != 0)
    //     max_nnz_per_dpu += ((8 / byte_dt) - (max_nnz_per_dpu % (8 / byte_dt)));
    // if (max_cols_per_tasklet % (8 / byte_dt) != 0)
    //     max_cols_per_tasklet += ((8 / byte_dt) - (max_cols_per_tasklet % (8 / byte_dt)));

    //maybe reallocation is needed
    
    //sending data to dpus

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // input_args[i].max_rows_per_tasklet = max_rows_per_tasklet; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    
    startTimer(&timer, 0);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, sparseA->nnzs));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "sparse", 0, sparseA->nnz * sizeof(struct elem_t), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0); 
    // printf("%d\n\n",sparseA->nnz);
    // printf("5\n");
    startTimer(&timer, 1);

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, denseB[i]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input",0 , m*sizeof(int8_t) , DPU_XFER_DEFAULT));//max_cols_per_dpu*m*sizeof(int8_t) +  sparseA->nnz * sizeof(struct elem_t)
    stopTimer(&timer, 1);
    // printf("6\n");
    startTimer(&timer, 2);
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    stopTimer(&timer, 2);
    // printf("7\n");
#if LOG
    // Display DPU Logs (default: disabled)
    i=0;
    DPU_FOREACH(dpu_set, dpu,i) {
        printf("\n______DPU: %d\n\n", i);
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif
    // printf("8\n");

    startTimer(&timer, 3);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, denseO[i]));
    } 
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "output",0, m* sizeof(int8_t), DPU_XFER_DEFAULT));
    stopTimer(&timer, 3);
    // printf("9\n");

    // init_denseMatrix(denseB, m, f);
    // for(int i=0; i<m; i++){
    //     for(int j=0; j<f;j++){
    //         printf(" %d ", denseO[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    printf("\n");
    printf("Load Sparse Matrix ");
    printTimer(&timer, 0);
    printf("Load Dense Matrix");
    printTimer(&timer, 1);
    printf("Kernel ");
    printTimer(&timer, 2);
    printf("Retrieve Output Matrix ");
    printTimer(&timer, 3);
    printf("\n\n");
    
    freeCOOMatrix(sparseA);
    // printf("1\n");
    
    // printf("2\n");
    for( i =0; i<f; i++){
        free(denseO[i]);
    }
     for(int i = 0; i<f; i++){
        free(denseB[i]);
    }
    // printf("3\n");
    free(denseB);
    // printf("4\n");
    free(denseO);
    // printf("5\n");
    // partition_free(part_info);
    // DPU_ASSERT(dpu_free(dpu_set));
    // free(dpu_info);
    // free(input_args);

    return 0;
}
























