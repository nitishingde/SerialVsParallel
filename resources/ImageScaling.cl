__kernel void nearestNeighbourInterpolation(
    __constant uchar *image,
    const size_t imageStride,
    __global uchar *scaledImage,
    const uint scaledImageRows,
    const uint scaledImageCols,
    const size_t scaledImageStride,
    const size_t channelSize,
    const float imageScaleX,
    const float imageScaleY
) {
    size_t scaledI = get_global_id(0), scaledJ = get_global_id(1);
    if(scaledImageRows <= scaledI || scaledImageCols <= scaledJ) return;
    size_t scaledImageOffset = scaledI*scaledImageStride + scaledJ*channelSize;

    size_t i = scaledI/imageScaleY, j = scaledJ/imageScaleX;
    size_t imageOffset = i*imageStride + j*channelSize;

    for(size_t channel = 0; channel < channelSize; ++channel) {
        scaledImage[scaledImageOffset + channel] = image[imageOffset + channel];
    }
}

__constant sampler_t nnSampler =    CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_NONE |
                                    CLK_FILTER_NEAREST;

__kernel void nearestNeighbourInterpolation2(
    __read_only image2d_t image,
    __write_only image2d_t scaledImage,
    const uint scaledImageRows,
    const uint scaledImageCols,
    const float imageScaleX,
    const float imageScaleY
) {
    // column major indexing
    int2 scaledImageCoord = (int2)(get_global_id(1), get_global_id(0));
    if(scaledImageRows <= scaledImageCoord.x || scaledImageCols <= scaledImageCoord.y) return;
    int2 imageCoord = (int2)(scaledImageCoord.x/imageScaleX, scaledImageCoord.y/imageScaleY);

    uint4 pixel = read_imageui(image, nnSampler, imageCoord);
    write_imageui(scaledImage, scaledImageCoord, pixel);
}
