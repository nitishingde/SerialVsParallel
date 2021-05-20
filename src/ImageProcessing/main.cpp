#include <memory>
#include "ImageProcessing.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    auto image = cv::imread("resources/2048x1024.png", cv::ImreadModes::IMREAD_UNCHANGED);
    if(image.channels() == 3) {
        cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2BGRA);
    }
    //FIXME
    //float scaleX = 4.f/3.f, scaleY = 5.f/3.f;
    float scaleX = 4.f, scaleY = 5.f;

    for(auto [pImageScalingStrategy, interpolationFlag]: {
        std::make_tuple(std::shared_ptr<svp::ImageScalingStrategy>(new svp::NNI_Serial()), cv::InterpolationFlags::INTER_NEAREST),
        std::make_tuple(std::shared_ptr<svp::ImageScalingStrategy>(new svp::NNI_OpenCL()), cv::InterpolationFlags::INTER_NEAREST),
        std::make_tuple(std::shared_ptr<svp::ImageScalingStrategy>(new svp::NNI_OpenCL2()), cv::InterpolationFlags::INTER_NEAREST),
    }) {
        cv::Mat expectedResult;
        cv::resize(image, expectedResult, cv::Size(), scaleX, scaleY, interpolationFlag);
        SVP_START_BENCHMARKING_SESSION(pImageScalingStrategy->toString().c_str(), 10) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto result = pImageScalingStrategy->transform(image, scaleX, scaleY);
#if not NDEBUG
            if(!svp::cmp(result, expectedResult)) {
                return -1;
            }
#endif
        }
    }

    return 0;
}