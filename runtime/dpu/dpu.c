#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <seqread.h>

#include "../support/common.h"
#include "../support/utils.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__mram_noinit int8_t input[MAX_ROWS]; 
__mram_noinit int8_t output[MAX_ROWS]; 
__mram_noinit struct elem_t sparse[MAX_NNZ]; 


BARRIER_INIT(my_barrier, NR_TASKLETS);
uint32_t nnz_offset;


int main() {
    uint32_t tasklet_id = me();

    if (tasklet_id == 0){ 
        mem_reset(); 
    }

    // Barrier
    barrier_wait(&my_barrier);

    uint32_t nrows = DPU_INPUT_ARGUMENTS.nrows;
    uint32_t cols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_cols_per_tasklet = DPU_INPUT_ARGUMENTS.max_rows_per_tasklet;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t tstart_row = DPU_INPUT_ARGUMENTS.tstart_row;
    uint32_t start_nnz = DPU_INPUT_ARGUMENTS.start_nnz[tasklet_id];
    uint32_t start_row = DPU_INPUT_ARGUMENTS.start_row[tasklet_id];
    uint32_t nnz = DPU_INPUT_ARGUMENTS.nnz;
    printf( "row, col, nnz: %d, %d,%d\n",nrows,cols,nnz);

    for(int i = 0; i<nrows ; i++){
        output[i]=0;
        for( int j = 0; j<nnz; j++){
            if( sparse[j].rowind == i){
                output[i] += sparse[j].val * input[i];
            }

        }
    }

}