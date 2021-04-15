#ifndef SERIALVSPARALLEL_IMAGEPROCESSING_H
#define SERIALVSPARALLEL_IMAGEPROCESSING_H


#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>

namespace svp {
    bool cmp(const cv::Mat &image1, const cv::Mat &image2);

    class ImageScalingStrategy {
    public:
        virtual cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) = 0;
        virtual std::string toString() = 0;
    };

    class NNI_Serial: public ImageScalingStrategy {
    public:
        cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) override;
        std::string toString() override;
    };

    class NNI_OpenCL: public ImageScalingStrategy {
    private:
        bool isInitialised = false;
        cl::Context mContext;
        cl::Device mDevice;
        cl::Kernel mKernel;
        cl::CommandQueue mCommandQueue;

    private:
        void init();
    public:
        explicit NNI_OpenCL();
        cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) override;
        std::string toString() override;
    };

    class NNI_OpenCL2: public ImageScalingStrategy {
    private:
        bool isInitialised = false;
        cl::Context mContext;
        cl::Device mDevice;
        cl::Kernel mKernel;
        cl::CommandQueue mCommandQueue;

    private:
        void init();
    public:
        explicit NNI_OpenCL2();
        cv::Mat transform(const cv::Mat &image, float scaleX, float scaleY) override;
        std::string toString() override;
    };

    class ImageScalingBenchMarker {
    private:
        std::unique_ptr<ImageScalingStrategy> mpImageScalingStrategy = nullptr;

    public:
        explicit ImageScalingBenchMarker(std::unique_ptr<ImageScalingStrategy> pImageScalingStrategy = nullptr);
        void setImageScalingStrategy(std::unique_ptr<ImageScalingStrategy> pImageScalingStrategy);
        void benchmarkImageScaling(uint32_t iterations, const cv::Mat &image, float scaleX, float scaleY, const cv::Mat &expectedResult) const;
    };
}


#endif //SERIALVSPARALLEL_IMAGEPROCESSING_H
