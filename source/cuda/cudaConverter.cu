#include "cudaConverter.h"
#include "cudaUtility.h"
#include "imageFormat.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <assert.h>

#include <stdio.h>

namespace cu {
    template<int inputNumChannels, int outputNumChannels, bool isInputBGR>
    static __global__ void convertIntPackedTo32FPlanar(unsigned char *input
        , int width
        , int height
        , int pitch
        , float* output
        , int* meanData
        , float* scales
        , int batch) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        if (outputNumChannels == 3) {
            for (int i = blockIdx.z; i < batch; i += gridDim.z) {
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    if (inputNumChannels == 1) {
                        //GRAY8 => RGB32F
                        output[i * width * height + width * height * k + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x) - meanData[k]) * scales[k];
                    }
                    else {
                        if (isInputBGR) {
                            //BGR(A)8 => RGB32F
                            output[i * width * height * 3 + width * height * k + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + (3 - 1 - k)) - meanData[k]) * scales[k];
                        }
                        else {
                            //RGB(A)8 => RGB32F
                            output[i * width * height * 3 + width * height * k + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + k) - meanData[k]) * scales[k];
                        }
                    }
                }
            }
        }
        else if (outputNumChannels == 1) {
            for (int i = blockIdx.z; i < batch; i += gridDim.z) {
                if (inputNumChannels == 1) {
                    //GRAY8 => GRAY32F
                    output[i * width * height + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x) - meanData[0]) * scales[0];
                }
                else {
                    //RGB -> Gray =>  B' = 0.299 R + 0.587 G + 0.114 B
                    if (isInputBGR) {
                        //BGR(A)8 => GRAY32F
                        output[i * width * height + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 2) * 0.299
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 1)) * 0.587
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 0)) * 0.114
                            - meanData[0]) * scales[0];
                    }
                    else {
                        //RGB(A)8 => GRAY32F
                        output[i * width * height + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 0) * 0.299
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 1)) * 0.587
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 2)) * 0.114
                            - meanData[0]) * scales[0];
                    }
                }
            }
        }
    }

    cudaError_t cudaConvert(unsigned char* input
        , int width
        , int height
        , int pitch
        , ImageFormat inputImageFormat
        , float* output
        , ImageFormat outputImageFormat
        , int* meanData
        , float* scales
        , int nBatchSize
        , cudaStream_t stream) {

        // Restricting blocks in Z-dim till 32 to not launch too many blocks
        assert(nBatchSize <= 32);

        if (width == 0 || height == 0 || pitch == 0)
            return cudaErrorInvalidValue;

        // launch kernel
        const dim3 blockDim(32, 32, 1);
        const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), nBatchSize);

        if (outputImageFormat == ImageFormat::IMAGE_RGB32F_PLANAR){
            switch (inputImageFormat) {
            case ImageFormat::IMAGE_RGB8:
                convertIntPackedTo32FPlanar<3, 3, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_BGR8:
                convertIntPackedTo32FPlanar<3, 3, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_RGBA8:
                convertIntPackedTo32FPlanar<4, 3, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_BGRA8:
                convertIntPackedTo32FPlanar<4, 3, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_GRAY8:
                convertIntPackedTo32FPlanar<1, 3, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            default:
                return cudaErrorInvalidValue;
            }
        }
        else if (outputImageFormat == ImageFormat::IMAGE_GRAY32F) {
            switch (inputImageFormat){
            case ImageFormat::IMAGE_RGB8:
                convertIntPackedTo32FPlanar<3, 1, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_BGR8:
                convertIntPackedTo32FPlanar<3, 1, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_BGRA8:
                convertIntPackedTo32FPlanar<4, 1, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_GRAY8:
                convertIntPackedTo32FPlanar<1, 1, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, meanData, scales, nBatchSize);
                break;
            default:
                return cudaErrorInvalidValue;
            }
        }
        else {
            return cudaErrorInvalidValue;
        }
        return CUDA(cudaGetLastError());
    }

    /*
    * Resize and convert inside one cuda kernel
    */
    template<int inputNumChannels, int outputNumChannels, bool isInputBGR>
    static __global__ void resizeConvertIntPackedTo32FPlanar(unsigned char *input
        , int iWidth
        , int iHeight
        , int iPitch
        , float2 resize_scale
        , float* output
        , int oWidth
        , int oHeight
        , int* meanData
        , float* scales
        , int batch) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= iWidth || y >= iHeight)
            return;

        const int dx = ((float)x * resize_scale.x);
        const int dy = ((float)y * resize_scale.y);

        if (outputNumChannels == 3) {
            for (int i = blockIdx.z; i < batch; i += gridDim.z) {
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    if (inputNumChannels == 1) {
                        //GRAY8 => RGB32F_Planar
                        output[i * oWidth * oHeight + oWidth * oHeight * k + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx) - meanData[k]) * scales[k];
                    }
                    else {
                        if (isInputBGR) {
                            //BGR(A)8 => RGB32F_Planar
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * k + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + (3 - 1 - k)) - meanData[k]) * scales[k];
                        }
                        else {
                            //RGB(A)8 => RGB32F_Planar
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * k + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + k) - meanData[k]) * scales[k];
                        }
                    }
                }
            }
        }
    }

    cudaError_t cudaResizeConvert(unsigned char* input
        , int inputWidth
        , int inputHeight
        , int inputPitch
        , ImageFormat inputImageFormat
        , float* output
        , int outputWidth
        , int outputHeight
        , ImageFormat outputImageFormat
        , int* meanData
        , float* scales
        , int nBatchSize
        , cudaStream_t stream) {

        // Restricting blocks in Z-dim till 32 to not launch too many blocks
        assert(nBatchSize <= 32);

        if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 || inputPitch == 0)
            return cudaErrorInvalidValue;

        const float2 resize_scale = make_float2(float(inputWidth) / float(outputWidth),
            float(inputHeight) / float(outputHeight));

        // launch kernel
        const dim3 blockDim(32, 32, 1);
        const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y), nBatchSize);

        if (outputImageFormat == ImageFormat::IMAGE_RGB32F_PLANAR){
            switch (inputImageFormat) {
            case ImageFormat::IMAGE_RGB8:
                resizeConvertIntPackedTo32FPlanar<3, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_BGR8:
                resizeConvertIntPackedTo32FPlanar<3, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_RGBA8:
                resizeConvertIntPackedTo32FPlanar<4, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_BGRA8:
                resizeConvertIntPackedTo32FPlanar<4, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, meanData, scales, nBatchSize);
                break;
            case ImageFormat::IMAGE_GRAY8:
                resizeConvertIntPackedTo32FPlanar<1, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, meanData, scales, nBatchSize);
                break;
            default:
                return cudaErrorInvalidValue;
            }
        }
        else{
            return cudaErrorInvalidValue;
        }

        return CUDA(cudaGetLastError());
    }
}