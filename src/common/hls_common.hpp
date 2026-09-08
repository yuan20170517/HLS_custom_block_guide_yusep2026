#pragma once

#include <cstdint>

// Compatibility layer for AMD Vitis HLS fixed-width types and standard C++ simulation
#if defined(__SYNTHESIS__) || defined(HLS_NO_XIL_FPO_LIB)
#include <ap_int.h>
typedef ap_int<8>   int8_t_hls;
typedef ap_uint<8>  uint8_t_hls;
#else
#if defined(__has_include)
  #if __has_include(<ap_int.h>)
    #include <ap_int.h>
    typedef ap_int<8>   int8_t_hls;
    typedef ap_uint<8>  uint8_t_hls;
  #else
    typedef int8_t   int8_t_hls;
    typedef uint8_t  uint8_t_hls;
  #endif
#else
typedef int8_t   int8_t_hls;
typedef uint8_t  uint8_t_hls;
#endif
#endif
