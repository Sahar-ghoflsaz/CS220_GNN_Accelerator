#ifndef _UTILS_H_
#define _UTILS_H_

#define MAX_ROWS 19740
#define MAX_NNZ 44338
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define LOG 0
#define CHECK_CORR 0

// Define datatype for matrix elements
#if INT8
typedef int8_t val_dt;
#define byte_dt 1
#elif INT16
typedef int16_t val_dt;
#define byte_dt 2
#elif INT32
typedef int32_t val_dt;
#define byte_dt 4
#elif INT64
typedef int64_t val_dt;
#define byte_dt 8
#elif FP32 
typedef float val_dt;
#define byte_dt 4
#elif FP64
typedef double val_dt;
#define byte_dt 8
#else
typedef int32_t val_dt;
#define byte_dt 4
#endif

/**
 * @brief nnz in COO matrix format 
 */
struct elem_t {
    uint16_t rowind;
    uint16_t colind;
    int32_t val;
};


#endif