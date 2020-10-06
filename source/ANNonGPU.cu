#include "types.h"
#include <cuda_profiler_api.h>


namespace ann_on_gpu {

void setDevice(int device) {
    cudaSetDevice(device);
}

void start_profiling() {
    cudaProfilerStart();
}

void stop_profiling() {
    cudaProfilerStop();
}

} // namespace ann_on_gpu
