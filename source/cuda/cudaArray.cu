#include "cudaArray.h"
#include "cudaUtility.h"

namespace cu {
    template<typename T>
    __global__ void fillKernel(T * devPtr, const T val, const size_t size){
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (; tidx < size; tidx += stride)
            devPtr[tidx] = val;
    }

    template<typename T>
    cudaError_t cudaArrayFillValue(T* array, const T value, size_t size, cudaStream_t stream) {
        if (array == nullptr || size == 0) {
            return cudaErrorInvalidValue;
        }

        const dim3 blockDim(512);
        const dim3 gridDim(iDivUp(size, blockDim.x));

        fillKernel << <gridDim, blockDim, 0, stream >> >(array, value, size);
        return CUDA(cudaGetLastError());
    }

    template cudaError_t cudaArrayFillValue(float* array, const float value, size_t size, cudaStream_t stream);
}