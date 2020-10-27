/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#ifndef __IMAGE_FORMAT_H_
#define __IMAGE_FORMAT_H_

#include "cuda_utils_sdk.h"

// include vector types (float4, float3, uchar4, uchar3, ect.)
#include "cudaUtility.h"		

namespace cu {
    /**
     * The ImageFormat enum is used to identify the pixel format and colorspace
     * of an image.  Supported data types are based on `uint8` and `float`, with
     * colorspaces including RGB/RGBA, BGR/BGRA, grayscale, YUV, and Bayer.
     *
     * There are also a variety of helper functions available that provide info about
     * each format at runtime - for example, the pixel bit depth (ImageFormat())
     * the number of image channels (ImageFormat()), and computing the size of
     * an image from it's dimensions (@see ImageFormat()).  To convert between
     * image formats using the GPU, there is also the cudaConvertColor() function.
     *
     * In addition to the enums below, each format can also be identified by a string.
     * The string corresponding to each format is included in the documentation below.
     * These strings are more commonly used from Python, but can also be used from C++
     * with the imageFormatFromStr() and ImageFormat() functions.
     *
     * @ingroup imageFormat
     */

    /**
     * The imageBaseType enum is used to identify the base data type of an
     * ImageFormat - either uint8 or float.  For example, the IMAGE_RGB8
     * format has a base type of uint8, while IMAGE_RGB32F is float.
     *
     * You can retrieve the base type of each format with imageFormatBaseType()
     *
     * @ingroup imageFormat
     */
    enum imageBaseType
    {
        IMAGE_UINT8,
        IMAGE_FLOAT
    };

    /**
     * Get the base type of an image format (uint8 or float).
     * @see imageBaseType
     * @ingroup imageFormat
     */
    inline imageBaseType imageFormatBaseType(ImageFormat format);

    /**
     * Convert an ImageFormat enum to a string.
     * @see ImageFormat for the strings that correspond to each format.
     * @ingroup imageFormat
     */
    inline const char* imageFormatToStr(ImageFormat format);

    /**
     * Parse an ImageFormat enum from a string.
     * @see ImageFormat for the strings that correspond to each format.
     * @returns the ImageFormat, or IMAGE_UNKNOWN on an unrecognized string.
     * @ingroup imageFormat
     */
    inline ImageFormat imageFormatFromStr(const char* str);

    /**
     * Get the number of image channels in each format.
     * For example, IMAGE_RGB8 has 3 channels, while IMAGE_RGBA8 has 4.
     * @ingroup imageFormat
     */
    inline size_t imageFormatChannels(ImageFormat format);

    /**
     * Get the pixel bit depth (in bits, not bytes).
     *
     * The bit depth is the size in bits of each pixel in the image. For example,
     * IMAGE_RGB8 has a bit depth of 24.  This function returns bits instead of bytes,
     * because some formats have a bit depth that's not evenly divisible by 8 (a byte).
     * YUV 4:2:0 formats like I420, YV12, and NV12 have a depth of 12 bits.
     *
     * If you are calculating the overall size of an image, it's recommended to use
     * the imageFormatSize() function instead.  It will automatically convert to bytes.
     *
     * @ingroup imageFormat
     */
    inline size_t imageFormatDepth(ImageFormat format);

    /**
     * Compute the size of an image (in bytes)
     * @ingroup imageFormat
     */
    inline size_t imageFormatSize(ImageFormat format, size_t width, size_t height);

    /**
     * Check if an image format is one of the RGB/RGBA formats.
     *
     * @returns true if the ImageFormat is a RGB/RGBA format
     *               (IMAGE_RGB8, IMAGE_RGBA8, IMAGE_RGB32F, IMAGE_RGBA32F)
     *               otherwise, returns false.
     * @ingroup imageFormat
     */
    inline bool imageFormatIsRGB(ImageFormat format);

    /**
     * Check if an image format is one of the BGR/BGRA formats.
     *
     * @returns true if the ImageFormat is a BGR/BGRA format
     *               (IMAGE_BGR8, IMAGE_BGRA8, IMAGE_BGR32F, IMAGE_BGRA32F)
     *               otherwise, returns false.
     * @ingroup imageFormat
     */
    inline bool imageFormatIsBGR(ImageFormat format);

    /**
     * Check if an image format is one of the YUV formats.
     *
     * @returns true if the imageFormat is a YUV format
     *               (IMAGE_YUYV, IMAGE_YVYU, IMAGE_UYVY, IMAGE_I420, IMAGE_YV12, IMAGE_NV12)
     *               otherwise, returns false.
     * @ingroup imageFormat
     */
    inline bool imageFormatIsYUV(ImageFormat format);

    /**
     * Check if an image format is one of the grayscale formats.
     *
     * @returns true if the imageFormat is grayscale (IMAGE_GRAY8, IMAGE_GRAY32)
     *               otherwise, returns false.
     * @ingroup imageFormat
     */
    inline bool imageFormatIsGray(ImageFormat format);

    /**
     * Check if an image format is one of the Bayer formats.
     *
     * @returns true if the imageFormat is a Bayer format
     *               (IMAGE_BAYER_BGGR, IMAGE_BAYER_GBRG, IMAGE_BAYER_GRBG, IMAGE_BAYER_RGGB)
     *               otherwise, returns false.
     * @ingroup imageFormat
     */
    inline bool imageFormatIsBayer(ImageFormat format);


    //////////////////////////////////////////////////////////////////////////////////
    /// @name Internal Type Templates
    /// @internal
    /// @ingroup imageFormat
    //////////////////////////////////////////////////////////////////////////////////

    ///@{

    // get the IMAGE_RGB* formats from uchar3/uchar4/float3/float4
    template<typename T> inline ImageFormat imageFormatFromType();

    template<> inline ImageFormat imageFormatFromType<uchar3>();
    template<> inline ImageFormat imageFormatFromType<uchar4>();
    template<> inline ImageFormat imageFormatFromType<float3>();
    template<> inline ImageFormat imageFormatFromType<float4>();

    // templated version of base type / vector type
    template<ImageFormat format> struct imageFormatType;

    template<> struct imageFormatType<IMAGE_RGB8>    { typedef uint8_t Base; typedef uchar3 Vector; };
    template<> struct imageFormatType<IMAGE_RGBA8>   { typedef uint8_t Base; typedef uchar4 Vector; };

    template<> struct imageFormatType<IMAGE_RGB32F>  { typedef float Base; typedef float3 Vector; };
    template<> struct imageFormatType<IMAGE_RGBA32F> { typedef float Base; typedef float4 Vector; };
}

///@}

// inline implementations
#include "imageFormat.inl"

#endif
