#ifndef __CUDA_ARRAY_H__
#define __CUDA_ARRAY_H__

namespace cu {
    //!
    //! \brief Class cudaArrayFillValue uses CUDA to do array filling.
    //!
    //! \param array - gpu buffer
    //!
    //! \param value - value to be filled for all elements
    //!
    //! \param size - array size
    //!
    //! \param stream - stream used to do cuda operation
    //!
    template<typename T>
    cudaError_t cudaArrayFillValue(T* array, const T value, size_t size, cudaStream_t stream);
}

#endif