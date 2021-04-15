#include "ImageProcessing.h"


int main(int argc, char **argv) {
    auto image = cv::imread("2048x1024.png", cv::ImreadModes::IMREAD_UNCHANGED);
    if(image.channels() == 3) {
        cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2BGRA);
    }
    //FIXME
    //float scaleX = 4.f/3.f, scaleY = 5.f/3.f;
    float scaleX = 4.f, scaleY = 5.f;

    svp::ImageScalingBenchMarker imageScalingBenchMarker;
    for(auto [pImageScalingStrategy, interpolationFlag]: {
        std::make_tuple(static_cast<svp::ImageScalingStrategy*>(new svp::NNI_Serial()), cv::InterpolationFlags::INTER_NEAREST),
        std::make_tuple(static_cast<svp::ImageScalingStrategy*>(new svp::NNI_OpenCL()), cv::InterpolationFlags::INTER_NEAREST),
        std::make_tuple(static_cast<svp::ImageScalingStrategy*>(new svp::NNI_OpenCL2()), cv::InterpolationFlags::INTER_NEAREST),
    }) {
        imageScalingBenchMarker.setImageScalingStrategy(std::unique_ptr<svp::ImageScalingStrategy>(pImageScalingStrategy));
        cv::Mat expectedResult;
        cv::resize(image, expectedResult, cv::Size(), scaleX, scaleY, interpolationFlag);
        imageScalingBenchMarker.benchmarkImageScaling(10, image, scaleX, scaleY, expectedResult);
    }

    return 0;
}