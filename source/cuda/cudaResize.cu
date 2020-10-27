#include "cudaResize.h"

namespace cu {
    template<typename T, int numChannel, bool isPlanar>
    __global__ void gpuRGBLikeResize(T* input, int iWidth, int iHeight, int pitch, float2 scale, T* output, int oWidth, int oHeight)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= oWidth || y >= oHeight)
            return;

        const int dx = ((float)x * scale.x);
        const int dy = ((float)y * scale.y);

        if (isPlanar) {
#pragma unroll
            for (int k = 0; k < numChannel; ++k){
                output[oWidth * oHeight * k + y * oWidth + x] = input[dx * dy * k + dy * iWidth + dx];
            }
        }
        else {
#pragma unroll
            for (int k = 0; k < numChannel; ++k){
                output[y * oWidth * numChannel + x * numChannel + k] = input[dy * pitch + dx * numChannel + k];
            }
        }
    }

    template<typename T>
    cudaError_t cudaResizeRGBLike(T* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , T* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream) {
        if (!input || !output)
            return cudaErrorInvalidDevicePointer;

        if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 || pitch == 0)
            return cudaErrorInvalidValue;

        const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
            float(inputHeight) / float(outputHeight));

        // launch kernel
        const dim3 blockDim(32, 32);
        const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y));

        switch (inputImageFormat) {
        case ImageFormat::IMAGE_RGB8:
        case ImageFormat::IMAGE_BGR8:
            gpuRGBLikeResize<T, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight);
            break;
        case ImageFormat::IMAGE_RGBA8:
        case ImageFormat::IMAGE_BGRA8:
            gpuRGBLikeResize<T, 4, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight);
            break;
        case ImageFormat::IMAGE_GRAY8:
            gpuRGBLikeResize<T, 1, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight);
            break;
        case ImageFormat::IMAGE_RGB32F_PLANAR:
            gpuRGBLikeResize<T, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight);
            break;
        default:
            return cudaErrorInvalidValue;
        }

        return CUDA(cudaGetLastError());
    }

    template cudaError_t cudaResizeRGBLike(unsigned char* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , unsigned char* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);

    template cudaError_t cudaResizeRGBLike(float* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);
}