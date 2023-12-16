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
static int8_t** denseO_host;
#define FEATURE 2048

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

    //this matrix is coming from the python code and the structure is like row, col, val
    denseB = (int8_t**) malloc(f*sizeof(int8_t*));
    for( unsigned int i = 0; i<f; i++ ){
        denseB[i] = (int8_t*) malloc(m*sizeof(int8_t));
    }

    denseO = (int8_t**) malloc(f*sizeof(int8_t*));
    for( unsigned int i = 0; i<f; i++ ){
        denseO[i] = (int8_t*) malloc(m*sizeof(int8_t));
    }
    denseO_host = (int8_t**) malloc(f*sizeof(int8_t*));
    for( unsigned int i = 0; i<f; i++ ){
        denseO_host[i] = (int8_t*) malloc(m*sizeof(int8_t));
    }
    //initialize matrix b
    init_denseMatrix(denseB, m, f);
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

        uint32_t cols_per_dpu= 1;
        uint32_t prev_cols_dpu = 1;
        
        dpu_info[i].cols_per_dpu= cols_per_dpu;
        dpu_info[i].prev_cols_dpu = prev_cols_dpu;

        input_args[i].nrows = m;
        input_args[i].tcols = 1; 
        input_args[i].nnz = sparseA->nnz;

    }
    
    //cpu computation
    startTimer(&timer, 5);
    for(int k = 0; k<FEATURE ; k++){
        for(int i = 0; i<sparseA->nrows ; i++){
        denseO_host[k][i]=0;
        for( int j = 0; j<nnz; j++){
            if( sparseA->nnzs[j].rowind == i){
                denseO_host[k][i] += sparseA->nnzs[j].val * denseB[k][i];
            }
        }
        }
    }
    stopTimer(&timer, 5); 
    printf("CPU Computation: ");
    printTimer(&timer, 5);
    printf("\n\n");
    //sending data to dpus

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
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

    startTimer(&timer, 1);

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, denseB[i]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input",0 , m*sizeof(int8_t) , DPU_XFER_DEFAULT));//max_cols_per_dpu*m*sizeof(int8_t) +  sparseA->nnz * sizeof(struct elem_t)
    stopTimer(&timer, 1);
    startTimer(&timer, 2);
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    stopTimer(&timer, 2);
#if LOG
    i=0;
    DPU_FOREACH(dpu_set, dpu,i) {
        printf("\n______DPU: %d\n\n", i);
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

    startTimer(&timer, 3);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, denseO[i]));
    } 
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "output",0, m* sizeof(int8_t), DPU_XFER_DEFAULT));
    stopTimer(&timer, 3);

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

    for( i =0; i<f; i++){
        free(denseO[i]);
    }
     for(int i = 0; i<f; i++){
        free(denseB[i]);
    }
    for( i =0; i<f; i++){
        free(denseO_host[i]);
    }
    free(denseB);
    free(denseO);
    free(denseO_host);

    return 0;
}
























