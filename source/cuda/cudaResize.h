#ifndef __CUDA_RESIZE_H__
#define __CUDA_RESIZE_H__

#include "cudaUtility.h"
#include "imageFormat.h"

namespace cu{
    /**
    * Function for increasing or decreasing the size of an image of like RGB on the GPU,
    * etc, RGB8, RGB32F, BGR8 BGR32F, RGBA8, RGBA32F, BGRA8, BGRA32F, and those corresponding planar format
    * @ingroup cuda
    */
    template<typename T>
    cudaError_t cudaResizeRGBLike(T* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , T* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);



        /**
    * Function for padding increasing or padding decreasing the size of an image of like RGB on the GPU ,
    * etc, RGB8, RGB32F, BGR8 BGR32F, RGBA8, RGBA32F, BGRA8, BGRA32F, and those corresponding planar format
    * @ingroup cuda
    */
    template<typename T>
    cudaError_t cudaResizePaddingRGBLike(T* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , T* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);
}

#endif

