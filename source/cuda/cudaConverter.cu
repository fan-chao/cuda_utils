#include "cudaConverter.h"
#include "cudaUtility.h"
#include "imageFormat.h"


#include <cuda.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>

#include<algorithm>
#include<math.h>
#include <fstream>
#include <iostream>

namespace cu {
    template<int inputNumChannels, int outputNumChannels, bool isInputBGR>
    static __global__ void convertIntPackedTo32FPlanar(const unsigned char * __restrict__ input
        , int width
        , int height
        , int pitch
        , float* __restrict__ output
        , int batchSize
        , float3 meanData
        , float scale) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        if (outputNumChannels == 3) {
            for (int i = blockIdx.z; i < batchSize; i += gridDim.z) {
                if (inputNumChannels == 1) {
                    //GRAY8 => RGB32F
                    output[i * width * height + width * height * 0 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x)) * scale - meanData.x;
                    output[i * width * height + width * height * 1 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x)) * scale - meanData.y;
                    output[i * width * height + width * height * 2 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x)) * scale - meanData.z;
                }
                else {
                    if (isInputBGR) {
                        //BGR(A)8 => RGB32F
                        output[i * width * height * 3 + width * height * 0 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 2)) * scale - meanData.x;
                        output[i * width * height * 3 + width * height * 1 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 1)) * scale - meanData.y;
                        output[i * width * height * 3 + width * height * 2 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 0)) * scale - meanData.z;
                    }
                    else {
                        //RGB(A)8 => RGB32F
                        output[i * width * height * 3 + width * height * 0 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 0)) * scale - meanData.x;
                        output[i * width * height * 3 + width * height * 1 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 1)) * scale - meanData.y;
                        output[i * width * height * 3 + width * height * 2 + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 2)) * scale - meanData.z;
                    }
                }
            }
        }
        else if (outputNumChannels == 1) {
            for (int i = blockIdx.z; i < batchSize; i += gridDim.z) {
                if (inputNumChannels == 1) {
                    //GRAY8 => GRAY32F
                    output[i * width * height + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x)) * scale - meanData.x;
                }
                else {
                    //RGB -> Gray =>  B' = 0.299 R + 0.587 G + 0.114 B
                    if (isInputBGR) {
                        //BGR(A)8 => GRAY32F
                        output[i * width * height + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 2) * 0.299
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 1)) * 0.587
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 0)) * 0.114) * scale
                            - meanData.x;
                    }
                    else {
                        //RGB(A)8 => GRAY32F
                        output[i * width * height + y * width + x] = (float)(*(input + i * height * pitch + y * pitch + x * inputNumChannels + 0) * 0.299
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 1)) * 0.587
                            + (*(input + i * height * pitch + y * pitch + x * inputNumChannels + 2)) * 0.114) * scale
                            - meanData.x;
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
        , int batchSize
        , float* meanData
        , float scale
        , cudaStream_t stream) {

        if (width == 0 || height == 0 || pitch == 0)
            return cudaErrorInvalidValue;

        const float3 mean_data = make_float3(meanData[0], meanData[1], meanData[2]);

        // launch kernel
        const dim3 blockDim(32, 32, 1);
        const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), batchSize);

        if (outputImageFormat == ImageFormat::IMAGE_RGB32F_PLANAR){
            switch (inputImageFormat) {
            case ImageFormat::IMAGE_RGB8:
                convertIntPackedTo32FPlanar<3, 3, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGR8:
                convertIntPackedTo32FPlanar<3, 3, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_RGBA8:
                convertIntPackedTo32FPlanar<4, 3, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGRA8:
                convertIntPackedTo32FPlanar<4, 3, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_GRAY8:
                convertIntPackedTo32FPlanar<1, 3, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            default:
                return cudaErrorInvalidValue;
            }
        }
        else if (outputImageFormat == ImageFormat::IMAGE_GRAY32F) {
            switch (inputImageFormat){
            case ImageFormat::IMAGE_RGB8:
                convertIntPackedTo32FPlanar<3, 1, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGR8:
                convertIntPackedTo32FPlanar<3, 1, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGRA8:
                convertIntPackedTo32FPlanar<4, 1, true> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_GRAY8:
                convertIntPackedTo32FPlanar<1, 1, false> << <gridDim, blockDim, 0, stream >> >(input, width, height, pitch
                    , output, batchSize, mean_data, scale);
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
    static __global__ void resizeConvertIntPackedTo32FPlanar(const unsigned char * __restrict__ input
        , int iWidth
        , int iHeight
        , int iPitch
        , float2 resizeScale
        , float* __restrict__ output
        , int oWidth
        , int oHeight
        , int batchSize
        , float3 meanData
        , float scale) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;//cuda线程的x索引.x对应的comuln.这里的x y是相对输出来说的,是指输出的宽高,不是输入的.
        const int y = blockIdx.y * blockDim.y + threadIdx.y;//cuda线程的y索引.y对应的是row.
        
        //对于padding来说,x,y还是要用包含黑边的那个大的宽高,只是下面求dx dy的时候用不包含黑边的x,y.

        if (x >= iWidth || y >= iHeight)
            return;
       

        //resizeScale是在外面求出来的,是输入/输出.
        const int dx = ((float)x * resizeScale.x); //resizeScale.x = (float(inputWidth) / float(outputWidth);//假设输入是10*10,输出是5*5,那么dx=2x,dx其实是指输入的,x是指输出的.
        const int dy = ((float)y * resizeScale.y);//resizeScale.y = (float(inputHeight) / float(outputHeight));


        if (outputNumChannels == 3) 
        {
            for (int i = blockIdx.z; i < batchSize; i += gridDim.z)//blockIdx.z=1,  gridDim.z=batchsize.
            {
                if (inputNumChannels == 1)//输入是单通道的.
                {
                    //GRAY8 => RGB32F_Planar
                    output[i * oWidth * oHeight + oWidth * oHeight * 0 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx)) * scale - meanData.x;
                    output[i * oWidth * oHeight + oWidth * oHeight * 1 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx)) * scale - meanData.y;
                    output[i * oWidth * oHeight + oWidth * oHeight * 2 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx)) * scale - meanData.z;
                }
                else//输入是3通道或4通道.
                {
                    if (isInputBGR)//BGR 
                    { 
                        output[i * oWidth * oHeight * 3 + oWidth * oHeight * 0 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 2)) * scale - meanData.x;
                        output[i * oWidth * oHeight * 3 + oWidth * oHeight * 1 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 1)) * scale - meanData.y;
                        output[i * oWidth * oHeight * 3 + oWidth * oHeight * 2 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 0)) * scale - meanData.z;

                    }
                    else//RGB 
                    {
                        //RGB(A)8 => RGB32F_Planar
                        output[i * oWidth * oHeight * 3 + oWidth * oHeight * 0 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 0)) * scale - meanData.x;
                        output[i * oWidth * oHeight * 3 + oWidth * oHeight * 1 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 1)) * scale - meanData.y;
                        output[i * oWidth * oHeight * 3 + oWidth * oHeight * 2 + y * oWidth + x] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 2)) * scale - meanData.z;
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
        , int batchSize
        , float* meanData
        , float scale
        , cudaStream_t stream) {
        // Restricting blocks in Z-dim till 32 to not launch too many blocks
        assert(batchSize <= 32);

        if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 || inputPitch == 0)
            return cudaErrorInvalidValue;

        const float2 resize_scale = make_float2(float(inputWidth) / float(outputWidth),
        float(inputHeight) / float(outputHeight));


        const float3 mean_data = make_float3(meanData[0], meanData[1], meanData[2]);

        // launch kernel
        const dim3 blockDim(32, 32, 1);
        const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y), batchSize);//grid的宽高这样求,是为了保证一个grid能够处理完一张图片.

        if (outputImageFormat == ImageFormat::IMAGE_RGB32F_PLANAR)
        {
            switch (inputImageFormat) {
            case ImageFormat::IMAGE_RGB8:
                resizeConvertIntPackedTo32FPlanar<3, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGR8:
                resizeConvertIntPackedTo32FPlanar<3, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_RGBA8:
                resizeConvertIntPackedTo32FPlanar<4, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGRA8:
                resizeConvertIntPackedTo32FPlanar<4, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_GRAY8:
                resizeConvertIntPackedTo32FPlanar<1, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, batchSize, mean_data, scale);
                break;
            default:
                return cudaErrorInvalidValue;
            }
            
        }
        else{
            return cudaErrorInvalidValue;
        }
        printf("===========function:%s,line:%d\n", __FUNCTION__, __LINE__);
        return CUDA(cudaGetLastError());
    }





    /*
    * Resize ,convert and padding inside one cuda kernel
    */
    template<int inputNumChannels, int outputNumChannels, bool isInputBGR>
    static __global__ void resizeConvertPaddingIntPackedTo32FPlanar(const unsigned char * __restrict__ input
        , int iWidth
        , int iHeight
        , int iPitch
        , float2 resizeScale
        , float* __restrict__ output
        , int oWidth
        , int oHeight
        , int padd_w
        , int padd_h 
        , int batchSize
        , float3 meanData
        , float scale) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;//cuda线程的x索引.x对应的comuln.这里的x y是相对输出来说的,是指输出的宽高,不是输入的.
        const int y = blockIdx.y * blockDim.y + threadIdx.y;//cuda线程的y索引.y对应的是row.
        
        //对于padding来说,x,y还是要用包含黑边的那个大的宽高,只是下面求dx dy的时候用不包含黑边的x,y.

        if (x >= iWidth || y >= iHeight)
            return;
       

        //resizeScale是在外面求出来的,是输入/输出.
        const int dx = ((float)x * resizeScale.x);     //resizeScale.x = (float(inputWidth) / float(outputWidth);//假设输入是10*10,输出是5*5,那么dx=2x,dx其实是指输入的,x是指输出的.
        const int dy = ((float)y * resizeScale.y);    //resizeScale.y = (float(inputHeight) / float(outputHeight));
        

        if (outputNumChannels == 3) 
        {
            for (int i = blockIdx.z; i < batchSize; i += gridDim.z)//blockIdx.z=1,  gridDim.z=batchsize.
            {
                if (inputNumChannels == 1)//输入是单通道的.
                {
                    //先把原图都采样,所以这里用 输入 的宽高进行判断,另外因为输出的上下左右要填充黑边,所以真实的输出要平移,所以x要加上padd_w,y要加上padd_h.
                    if((dx < iWidth) && (dy < iHeight))
                    {
                        //GRAY8 => RGB32F_Planar
                        output[i * oWidth * oHeight + oWidth * oHeight * 0 + (y+padd_h) * oWidth + x+padd_w] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx)) * scale - meanData.x;
                        output[i * oWidth * oHeight + oWidth * oHeight * 1 + (y+padd_h) * oWidth + x+padd_w] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx)) * scale - meanData.y;
                        output[i * oWidth * oHeight + oWidth * oHeight * 2 + (y+padd_h) * oWidth + x+padd_w] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx)) * scale - meanData.z;
                    }
                    //填充0.
                    if(x < padd_w || y < padd_h || x >= (oWidth-padd_w) ||  y >= (oHeight - padd_h))
                    {
                        output[i * oWidth * oHeight + oWidth * oHeight * 0 + y * oWidth + x] = 0;
                        output[i * oWidth * oHeight + oWidth * oHeight * 1 + y * oWidth + x] = 0;
                        output[i * oWidth * oHeight + oWidth * oHeight * 2 + y * oWidth + x] = 0;
                    }
                    //else if(x > (oWidth-padd_w) || y > (oHeight - padd_h))
                    
                }
                else//输入是3通道或4通道.
                {
                    if (isInputBGR)//BGR 
                    { 
                        //先把原图都采样,所以这里用 输入 的宽高进行判断,另外因为输出的上下都要填充黑边,所以真实的输出要平移,所以x要加上padd_w,y要加上padd_h.
                        if((dx < iWidth) && (dy < iHeight))
                        {
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 0 + (y+padd_h) * oWidth + (x+padd_w)] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 2)) * scale - meanData.x;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 1 + (y+padd_h) * oWidth + (x+padd_w)] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 1)) * scale - meanData.y;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 2 + (y+padd_h) * oWidth + (x+padd_w)] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 0)) * scale - meanData.z;
                        }
                        //填充0.
                        if(x < padd_w || y < padd_h || x >= (oWidth-padd_w) || y >= (oHeight - padd_h))
                        {
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 0 + y * oWidth + x] = 0;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 1 + y * oWidth + x] = 0;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 2 + y * oWidth + x] = 0;
                        }

                    }
                    else//RGB 
                    {
                        //先把原图都采样,所以这里用 输入 的宽高进行判断,另外因为输出的上下都要填充黑边,所以真实的输出要平移,所以x要加上padd_w,y要加上padd_h.
                        if((dx < iWidth) && (dy < iHeight))
                        {
                            //RGB(A)8 => RGB32F_Planar
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 0 + (y+padd_h) * oWidth + x+padd_w] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 0)) * scale - meanData.x;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 1 + (y+padd_h) * oWidth + x+padd_w] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 1)) * scale - meanData.y;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 2 + (y+padd_h) * oWidth + x+padd_w] = (float)(*(input + i * iHeight * iPitch + dy * iPitch + dx * inputNumChannels + 2)) * scale - meanData.z;
                        }
                        //填充0.
                        if(x < padd_w || y < padd_h || x >= (oWidth-padd_w) || y >= (oHeight - padd_h))
                        {
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 0 + y * oWidth + x] = 0;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 1 + y * oWidth + x] = 0;
                            output[i * oWidth * oHeight * 3 + oWidth * oHeight * 2 + y * oWidth + x] = 0;
                        }
                    }
                }
            }
        }
    }


    cudaError_t cudaResizeConvertPadding(unsigned char* input
        , int inputWidth
        , int inputHeight
        , int inputPitch
        , ImageFormat inputImageFormat
        , float* output
        , int outputWidth
        , int outputHeight
        , ImageFormat outputImageFormat
        , int batchSize
        , float* meanData
        , float scale
        , cudaStream_t stream) {
        // Restricting blocks in Z-dim till 32 to not launch too many blocks
        assert(batchSize <= 32);

        if (inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 || inputPitch == 0)
            return cudaErrorInvalidValue;


        float r = max(float(inputWidth) / float(outputWidth), float(inputHeight) / float(outputHeight));
        float2 resize_scale = make_float2(r, r);
        int inside_w = round(inputWidth/r);//这个是用比例进行resize之后的宽,
        int inside_h = round(inputHeight/r);//这个是用比例进行resize之后的高.
        float padd_w = outputWidth -  inside_w ;
        float padd_h = outputHeight - inside_h ;
        padd_w = padd_w/2;
        padd_h = padd_h/2;
        std::cout<<"inside_w:"<<inside_w<<",,inside_h:"<<inside_h<<std::endl;
        std::cout<<"padd_w:"<<padd_w<<",padd_h:"<<padd_h<<std::endl;

        //const float2 resize_scale = make_float2(float(inputWidth) / float(outputWidth),
        //float(inputHeight) / float(outputHeight));


        const float3 mean_data = make_float3(meanData[0], meanData[1], meanData[2]);

        // launch kernel
        const dim3 blockDim(32, 32, 1);
        const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y), batchSize);//grid的宽高这样求,是为了保证一个grid能够处理完一张图片.
        
        
        if (outputImageFormat == ImageFormat::IMAGE_RGB32F_PLANAR)
        {
            switch (inputImageFormat) {
            case ImageFormat::IMAGE_RGB8:
                resizeConvertPaddingIntPackedTo32FPlanar<3, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, padd_w, padd_h, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGR8:
                resizeConvertPaddingIntPackedTo32FPlanar<3, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, padd_w, padd_h, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_RGBA8:
                resizeConvertPaddingIntPackedTo32FPlanar<4, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, padd_w, padd_h, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_BGRA8:
                resizeConvertPaddingIntPackedTo32FPlanar<4, 3, true> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, padd_w, padd_h, batchSize, mean_data, scale);
                break;
            case ImageFormat::IMAGE_GRAY8:
                resizeConvertPaddingIntPackedTo32FPlanar<1, 3, false> << <gridDim, blockDim, 0, stream >> >(input, inputWidth, inputHeight, inputPitch, resize_scale
                    , output, outputWidth, outputHeight, padd_w, padd_h, batchSize, mean_data, scale);
                break;
            default:
                return cudaErrorInvalidValue;
            }
            
        }
        else{
            return cudaErrorInvalidValue;
        }
        printf("===========function:%s,line:%d\n", __FUNCTION__, __LINE__);
        return CUDA(cudaGetLastError());
    }
}