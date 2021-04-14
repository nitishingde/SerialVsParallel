#include "ImageProcessing.h"


int main(int argc, char **argv) {
    auto image = cv::imread("Lenna.png", cv::IMREAD_UNCHANGED);
    //FIXME
    //float scaleX = 4.f/3.f, scaleY = 5.f/3.f;
    float scaleX = 4.f, scaleY = 5.f;

    svp::ImageScalingBenchMarker imageScalingBenchMarker;
    for(auto [pImageScalingStrategy, interpolationFlag]: {
        std::make_tuple(static_cast<svp::ImageScalingStrategy*>(new svp::NNI_Serial()), cv::InterpolationFlags::INTER_NEAREST),
    }) {
        imageScalingBenchMarker.setImageScalingStrategy(std::unique_ptr<svp::ImageScalingStrategy>(pImageScalingStrategy));
        cv::Mat expectedResult;
        cv::resize(image, expectedResult, cv::Size(), scaleX, scaleY, interpolationFlag);
        imageScalingBenchMarker.benchmarkImageScaling(10, image, scaleX, scaleY, expectedResult);
    }
    return 0;
}