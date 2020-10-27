#include "cuda_utils_sdk.h"

#include "cudaMappedMemory.h"
#include "cudaConverter.h"
#include "cudaResize.h"
#include "cudaArray.h"

#include "cuda_runtime.h"

using namespace cu;

/*
* cuda alloc memory mapped
*/
CU_API int cuAllocMapped(void** cpu_ptr, void** gpu_ptr, size_t size) {
    return cudaAllocMapped(cpu_ptr, gpu_ptr, size);
}

CU_API int cuAllocMapped(void** ptr, size_t size) {
    return cudaAllocMapped(ptr, size);
}

CU_API int cuFreeMapped(void* ptr) {
    return cudaFreeHost(ptr);
}

/*
* cuda resize
*/
CU_API int cuResizeRGBLike(unsigned char* input, size_t input_width, size_t input_height, int pitch, ImageFormat input_image_format
    , unsigned char* output, size_t output_width, size_t output_height, void* stream) {
    return cudaResizeRGBLike(input, input_width, input_height, pitch, input_image_format, output, output_width, output_height, (cudaStream_t)stream);
}

CU_API int cuResizeRGBLike(float* input, size_t input_width, size_t input_height, int pitch, ImageFormat input_image_format
    , float* output, size_t output_width, size_t output_height, void* stream) {
    return cudaResizeRGBLike(input, input_width, input_height, pitch, input_image_format, output, output_width, output_height, (cudaStream_t)stream);
}

/*
* cuda convert
*/
CU_API int cuConvert(unsigned char* input, int width, int height, int pitch, ImageFormat input_image_format
    , float* output, ImageFormat output_image_format, int* mean_data, float* scales, int n_batch_size, void* stream) {
    return cudaConvert(input, width, height, pitch, input_image_format, output, output_image_format, mean_data, scales, n_batch_size, (cudaStream_t)stream);
}

/*
* cuda resize and convert
*/
CU_API int cuResizeConvert(unsigned char* input, int input_width, int input_height, int input_pitch, ImageFormat input_image_format
    , float* output, int output_width, int output_height, ImageFormat output_image_format
    , int* mean_data, float* scales, int n_batch_size, void* stream) {
    return cudaResizeConvert(input, input_width, input_height, input_pitch, input_image_format, output, output_width, output_height, output_image_format
        , mean_data, scales, n_batch_size, (cudaStream_t)stream);
}

/*
* cuda fill array value
*/
CU_API int cuArrayFillValue(float* array, float value, size_t size, void* stream) {
    return cudaArrayFillValue(array, value, size, (cudaStream_t)stream);
}

/*
* cuda Synchronize
*/

CU_API int cuStreamSynchronize(void* stream) {
    return cudaStreamSynchronize((cudaStream_t)stream);
}

/*
* error message
*/
CU_API const char * cuErrorMessage(int error_code) {
    return cudaGetErrorString(cudaError_t(error_code));
}
