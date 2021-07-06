#include "cudaResize.h"
#include <fstream>
#include <iostream>


namespace cu {
    template<typename T, int numChannel, bool isPlanar>
    __global__ void gpuRGBLikeResize(const T* __restrict__ input, int iWidth, int iHeight, int pitch, float2 scale, T* __restrict__ output, int oWidth, int oHeight)
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










    template<typename T, int numChannel, bool isPlanar>
    __global__ void gpuRGBLikeResizePadding(const T* __restrict__ input, int iWidth, int iHeight, int pitch, float2 scale, T* __restrict__ output, int oWidth, int oHeight, int padd_w, int padd_h)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= oWidth || y >= oHeight)
            return;

        const int dx = ((float)x * scale.x);
        const int dy = ((float)y * scale.y);

        if (isPlanar) 
        {
#pragma unroll
            for (int k = 0; k < numChannel; ++k)
            {
                //先把原图都采样,所以这里用 输入 的宽高进行判断,另外因为输出的上下左右要填充黑边,所以真实的输出要平移,所以x要加上padd_w,y要加上padd_h.
                if((dx < iWidth) && (dy < iHeight))
                {
                    output[oWidth * oHeight * k + (y+padd_h) * oWidth + (x+padd_w)] = input[dx * dy * k + dy * iWidth + dx];
                }
                //填充0.
                if(x < padd_w || y < padd_h || x >= (oWidth-padd_w) ||  y >= (oHeight - padd_h))
                {
                     output[oWidth * oHeight * k + y * oWidth + x] = 0;
                }    
            }
        }
        else 
        {
#pragma unroll
            for (int k = 0; k < numChannel; ++k)
            {
                //先把原图都采样,所以这里用 输入 的宽高进行判断,另外因为输出的上下左右要填充黑边,所以真实的输出要平移,所以x要加上padd_w,y要加上padd_h.
                if((dx < iWidth) && (dy < iHeight))
                {
                    output[(y+padd_h) * oWidth * numChannel + (x+padd_w) * numChannel + k] = input[dy * pitch + dx * numChannel + k];
                }
                //填充0.
                if(x < padd_w || y < padd_h || x >= (oWidth-padd_w) ||  y >= (oHeight - padd_h))
                {
                     output[y * oWidth * numChannel + x * numChannel + k] = 0;
                }
                
            }
        }
    }

    template<typename T>
    cudaError_t cudaResizePaddingRGBLike(T* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , T* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream) {
        if (!input || !output)
            return cudaErrorInvalidDevicePointer;

        if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 || pitch == 0)
            return cudaErrorInvalidValue;

        //const float2 scale = make_float2(float(inputWidth) / float(outputWidth),
            //float(inputHeight) / float(outputHeight));


        float r = max(float(inputWidth) / float(outputWidth), float(inputHeight) / float(outputHeight));
        float2 scale = make_float2(r, r);
        int inside_w = round(inputWidth/r);//这个是用比例进行resize之后的宽,
        int inside_h = round(inputHeight/r);//这个是用比例进行resize之后的高.
        float padd_w = outputWidth -  inside_w ;
        float padd_h = outputHeight - inside_h ;
        padd_w = padd_w/2;
        padd_h = padd_h/2;
        std::cout<<"inside_w:"<<inside_w<<",,inside_h:"<<inside_h<<std::endl;
        std::cout<<"padd_w:"<<padd_w<<",padd_h:"<<padd_h<<std::endl;


        // launch kernel
        const dim3 blockDim(32, 32);
        const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y));

        switch (inputImageFormat) {
        case ImageFormat::IMAGE_RGB8:
        case ImageFormat::IMAGE_BGR8:
            gpuRGBLikeResizePadding<T, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight,  padd_w,  padd_h);
            break;
        case ImageFormat::IMAGE_RGBA8:
        case ImageFormat::IMAGE_BGRA8:
            gpuRGBLikeResizePadding<T, 4, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight,  padd_w,  padd_h);
            break;
        case ImageFormat::IMAGE_GRAY8:
            gpuRGBLikeResizePadding<T, 1, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight,  padd_w,  padd_h);
            break;
        case ImageFormat::IMAGE_RGB32F_PLANAR:
            gpuRGBLikeResizePadding<T, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, pitch, scale, output
                , outputWidth, outputHeight,  padd_w,  padd_h);
            break;
        default:
            return cudaErrorInvalidValue;
        }

        return CUDA(cudaGetLastError());
    }


    template cudaError_t cudaResizePaddingRGBLike(unsigned char* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , unsigned char* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);

    template cudaError_t cudaResizePaddingRGBLike(float* input, size_t inputWidth, size_t inputHeight, int pitch, ImageFormat inputImageFormat
        , float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream);


}