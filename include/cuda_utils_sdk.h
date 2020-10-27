#ifndef __CUDA_UTILS_SDK_H_
#define __CUDA_UTILS_SDK_H_

#include <stdlib.h>

#ifdef __GNUC__
#define CU_API  extern
#endif

/*
 * The imageFormat enum is used to identify the pixel format and colorspace
 * of an image.  Supported data types are based on `uint8` and `float`, with
 * colorspaces including RGB/RGBA, BGR/BGRA, grayscale, YUV, and Bayer.
 */
enum ImageFormat {
    // RGB
    IMAGE_RGB8 = 0,             /**< uchar3 RGB8    (`'rgb8'`) */
    IMAGE_RGBA8,                /**< uchar4 RGBA8   (`'rgba8'`) */
    IMAGE_RGB32F,               /**< float3 RGB32F  (`'rgb32f'`) */
    IMAGE_RGBA32F,              /**< float4 RGBA32F (`'rgba32f'`) */
    IMAGE_RGB32F_PLANAR,                /**< float RGB32F Planar - RRRGGGBBB...(float...)  (`'rgb32f'`) */

    // BGR
    IMAGE_BGR8,             /**< uchar3 BGR8    (`'bgr8'`) */
    IMAGE_BGRA8,                /**< uchar4 BGRA8   (`'bgra8'`) */
    IMAGE_BGR32F,               /**< float3 BGR32F  (`'bgr32f'`) */
    IMAGE_BGRA32F,              /**< float4 BGRA32F (`'bgra32f'`) */

    // YUV
    IMAGE_YUYV,             /**< YUV YUYV 4:2:2 packed (`'yuyv'`) */
    IMAGE_YUY2 = IMAGE_YUYV,                /**< Duplicate of YUYV     (`'yuy2'`) */
    IMAGE_YVYU,             /**< YUV YVYU 4:2:2 packed (`'yvyu'`) */
    IMAGE_UYVY,             /**< YUV UYVY 4:2:2 packed (`'uyvy'`) */
    IMAGE_I420,             /**< YUV I420 4:2:0 planar (`'i420'`) */
    IMAGE_YV12,             /**< YUV YV12 4:2:0 planar (`'yv12'`) */
    IMAGE_NV12,             /**< YUV NV12 4:2:0 planar (`'nv12'`) */

    // Bayer
    IMAGE_BAYER_BGGR,               /**< 8-bit Bayer BGGR (`'bayer-bggr'`) */
    IMAGE_BAYER_GBRG,               /**< 8-bit Bayer GBRG (`'bayer-gbrg'`) */
    IMAGE_BAYER_GRBG,               /**< 8-bit Bayer GRBG (`'bayer-grbg'`) */
    IMAGE_BAYER_RGGB,               /**< 8-bit Bayer RGGB (`'bayer-rggb'`) */

    // grayscale
    IMAGE_GRAY8,                /**< uint8 grayscale  (`'gray8'`)   */
    IMAGE_GRAY32F,              /**< float grayscale  (`'gray32f'`) */

    // extras
    IMAGE_COUNT,                /**< The number of image formats */
    IMAGE_UNKNOWN = 999,                /**< Unknown/undefined format */
    IMAGE_DEFAULT = IMAGE_RGBA32F               /**< Default format (IMAGE_RGBA32F) */
};

/*
* cuda alloc memory mapped
*/
CU_API int cuAllocMapped(void** cpu_ptr, void** gpu_ptr, size_t size);
CU_API int cuAllocMapped(void** ptr, size_t size);

CU_API int cuFreeMapped(void* ptr);

/*
* cuda resize
*/
CU_API int cuResizeRGBLike(unsigned char* input, size_t input_width, size_t input_height, int pitch, ImageFormat input_image_format
    , unsigned char* output, size_t output_width, size_t output_height, void* stream);
CU_API int cuResizeRGBLike(float* input, size_t input_width, size_t input_height, int pitch, ImageFormat input_image_format
    , float* output, size_t output_width, size_t output_height, void* stream);

/*
* cuda convert
*/
CU_API int cuConvert(unsigned char* input, int width, int height, int pitch, ImageFormat input_image_format
    , float* output, ImageFormat output_image_format, int* mean_data, float* scales, int n_batch_size, void* stream);

/*
* cuda resize and convert
*/
CU_API int cuResizeConvert(unsigned char* input, int input_width, int input_height, int input_pitch, ImageFormat input_image_format
    , float* output, int output_width, int output_height, ImageFormat output_image_format
    , int* offsets, float* scales, int nBatchSize, void* stream);

/*
* cuda fill array value
*/
CU_API int cuArrayFillValue(float* array, float value, size_t size, void* stream);

/*
* cuda Synchronize
*/

CU_API int cuStreamSynchronize(void* stream);


/*
* error message
*/
CU_API const char * cuErrorMessage(int error_code);

#endif //! __CUDA_UTILS_SDK_H_