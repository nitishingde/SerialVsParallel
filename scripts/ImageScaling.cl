__kernel void nearestNeighbourInterpolation(
    __constant uchar *image,
    const size_t imageStride,
    __global uchar *scaledImage,
    const size_t scaledImageStride,
    const size_t channelSize,
    const float imageScaleX,
    const float imageScaleY
) {
    size_t scaledI = get_global_id(0), scaledJ = get_global_id(1);
    size_t scaledImageOffset = scaledI*scaledImageStride + scaledJ*channelSize;

    size_t i = scaledI/imageScaleY, j = scaledJ/imageScaleX;
    size_t imageOffset = i*imageStride + j*channelSize;

    for(size_t channel = 0; channel < channelSize; ++channel) {
        scaledImage[scaledImageOffset + channel] = image[imageOffset + channel];
    }
}
