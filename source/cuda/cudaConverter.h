#ifndef __CUDA_CONVERTER_H__
#define __CUDA_CONVERTER_H__

#include "cuda_utils_sdk.h"

namespace cu {
    //!
    //! \brief Class CUDAConverter uses CUDA to do ABGR32  packed(int) or thoer format to BGR/RGB/Gray planar(float) conversion.
    //!
    //! \param input - input buffer, type maybe uchar or float 
    //!
    //! \param width - width of the frame
    //!
    //! \param height - height of the frame
    //!
    //! \param pitch - number of bytes at each row
    //!
    //! \param input_image_format - format of input frame
    //!
    //! \param output - output buffer, type maybe uchar or float
    //!
    //! \param output_image_format - format of output frame
    //!
    //! \param meanData - mean value for inference
    //!
    //! \param scales - scale the float for following inference
    //!
    //! \param nBatchSize - batch size
    //!
    //! \param stream - stream used to do cuda operation
    //!
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
        , cudaStream_t stream);


    //!
    //! \brief Class CUDAResizeConverter uses CUDA to do rezie and BGR  packed(int) to RGB planar(float) conversion.
    //! \include offset and scale
    //!
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
        , cudaStream_t stream);

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
    , cudaStream_t stream);

}
#endif // !__CUDA_CONVERTER_H__