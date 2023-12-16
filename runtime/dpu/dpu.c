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
        mem_reset(); // Reset the heap
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

    // Find start addresses in MRAM
    // uint32_t mram_base_addr_output = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    // uint32_t mram_temp_addr_output;
    // uint32_t mram_base_addr_a = (uint32_t) (DPU_MRAM_HEAP_POINTER);//(uint32_t) (DPU_MRAM_HEAP_POINTER +(max_cols_per_dpu * nrows*sizeof(int8_t)));
    // uint32_t mram_base_addr_a_temp = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    // uint32_t mram_base_addr_b = (uint32_t) (mram_base_addr_a + (nnzs_pad * sizeof(struct nnz_t)));
    // mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER + (tasklet_id * max_rows_per_tasklet * sizeof(val_dt)));

     // Initialize sequential reader for nnzs
    // mram_base_addr_elems += sizeof(struct elem_t); //for tasklets needs some changes
    // seqreader_buffer_t cache_elems = seqread_alloc();
    // seqreader_t sr_elem;
    // struct elem_t *cur_elem = seqread_init(cache_elems, (__mram_ptr void *) mram_base_addr_a_temp, &sr_elem);
    // uint32_t prev_row = cur_elem->rowind;

    printf("hi\n");
    for(int i = 0; i<nrows ; i++){
        printf("input[%d] : %d\n", i, input[i]);
        output[i]=0;
        // mram_base_addr_a_temp = mram_base_addr_a;
        // seqreader_buffer_t cache_elems = seqread_alloc();
        // seqreader_t sr_elem;
        // struct elem_t *cur_elem = seqread_init(cache_elems, (__mram_ptr void *) mram_base_addr_a_temp, &sr_elem);
        // uint32_t prev_row = cur_elem->rowind;
        for( int j = 0; j<nnz; j++){
            
            printf("cur_elem: row: %d, col: %d, val:%d\n",sparse[j].rowind,sparse[j].colind,sparse[j].val);
            if( sparse[j].rowind == i){
                output[i] += sparse[j].val * input[i];
            }
            // else if( sparse[j].rowind < i ){
            //     continue;
            // }
            // else if( sparse[j].rowind > i ){
            //     break;
            // }
            // cur_elem = seqread_get(cur_elem, sizeof(struct elem_t), &sr_elem);

        }
        printf("output[%d] : %d\n", i, output[i]);
    }
    // for(i=0; i<nrows_per_tasklet; i++) {

}