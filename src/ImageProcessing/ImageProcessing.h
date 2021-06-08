#ifndef SERIALVSPARALLEL_IMAGEPROCESSING_H
#define SERIALVSPARALLEL_IMAGEPROCESSING_H


#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include "../Utility.h"

namespace svp {
    bool cmp(const cv::Mat &image1, const cv::Mat &image2);

    class ImageScalingStrategy {
    public:
        virtual ~ImageScalingStrategy() = default;
        virtual cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) = 0;
        virtual std::string toString() = 0;
    };

    class NNI_Serial: public ImageScalingStrategy {
    public:
        cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) override;
        std::string toString() override;
    };

    class NNI_OpenCL: public ImageScalingStrategy, private OpenCL_Base {
    private:
        cl::Kernel mKernel;

    private:
        void init() override;
    public:
        explicit NNI_OpenCL();
        cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) override;
        std::string toString() override;
    };

    class NNI_OpenCL2: public ImageScalingStrategy, private OpenCL_Base {
    private:
        cl::Kernel mKernel;

    private:
        void init() override;
    public:
        explicit NNI_OpenCL2();
        cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) override;
        std::string toString() override;
    };
}


#endif //SERIALVSPARALLEL_IMAGEPROCESSING_H
