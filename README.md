# cuda_utils
CUDA utils API

将cuda的一些常用操作封装成接口API，供应用程序调用。此库基于 https://github.com/dusty-nv/jetson-inference 修改。

支持的功能：
1. 内存管理 - cuAllocMapped
在Jetson系列的集成盒子上，CPU和GPU共用同一个内存地址，加速数据传输。
在PC端的显卡，CPU和GPU的内存地址可能不一致，需要分开使用CPU和GPU的指针。

2. 硬件支持的缩放与颜色空间转换操作 - cuResizeRGBLike
实现将ARGB/RGBA转换成BGR，并且同时实现缩放操作。

3. 颜色空间转换与数据类型转换操作 - cuConvert
在实现颜色空间转换的同时，将会图像由UCHAR类型转为Float类型，并且可以同时进行减均值操作。

4. 缩放、颜色空间转换、数据类型转换、减均值同时完成操作 - cuResizeConvert

5. 大数组填充操作 - cuArrayFillValue
对于大数组，使用CUDA快速完成填充操作，如初始化。

6. CUDA Stream同步操作 - cuStreamSynchronize
