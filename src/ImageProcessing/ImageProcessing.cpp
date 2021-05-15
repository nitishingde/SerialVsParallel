#include "ImageProcessing.h"
#include "../Utility.h"
#include <chrono>
#include <numeric>

bool svp::cmp(const cv::Mat &image1, const cv::Mat &image2) {
    return
        image1.rows == image2.rows
        and image1.cols == image2.cols
        and image1.channels() == image2.channels()
        and std::equal(image1.begin<uint8_t>(), image1.end<uint8_t>(), image2.begin<uint8_t>())
        ;
}

cv::Mat svp::NNI_Serial::transform(const cv::Mat &image, float scaleX, float scaleY) {
    SVP_PROFILE_FUNC();

    auto channelSize = image.channels();
    cv::Mat scaledImage(std::round(image.rows * scaleY), std::round(image.cols * scaleX), CV_8UC(channelSize));
    for(int32_t scaledI = 0; scaledI < scaledImage.rows; ++scaledI) {
        for(int32_t scaledJ = 0; scaledJ < scaledImage.cols; ++scaledJ) {
            //4 "8-bit" channels -> 32 bits
            scaledImage.at<uint32_t>(scaledI, scaledJ) = image.at<uint32_t>(scaledI/scaleY, scaledJ/scaleX);
        }
    }

    return scaledImage;
}

std::string svp::NNI_Serial::toString() {
    return "Scaling image using Nearest Neighbour Interpolation using serial code";
}

svp::NNI_OpenCL::NNI_OpenCL() {
    init();
}

void svp::NNI_OpenCL::init() {
    if(isInitialised) return;

    cl_int status = CL_SUCCESS;

    mContext = cl::Context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    svp::verifyOpenCL_Status(status);

    auto devices = mContext.getInfo<CL_CONTEXT_DEVICES>(&status);
    svp::verifyOpenCL_Status(status);
    mDevice = devices.front();

    cl::Program program(
        mContext,
        svp::readScript("ImageScaling.cl"),
        false,
        &status
    );
    svp::verifyOpenCL_Status(status);
    svp::verifyOpenCL_Status(program.build("-cl-std=CL1.2"));
    mKernel = cl::Kernel(program, "nearestNeighbourInterpolation", &status);
    svp::verifyOpenCL_Status(status);

    mCommandQueue = cl::CommandQueue(mContext, mDevice, 0, &status);
    svp::verifyOpenCL_Status(status);

    isInitialised = true;
}

cv::Mat svp::NNI_OpenCL::transform(const cv::Mat &image, float scaleX, float scaleY) {
    SVP_PROFILE_FUNC();

    cl_int status;
    cv::Mat scaledImage(std::round(image.rows * scaleY), std::round(image.cols * scaleX), CV_8UC(image.channels()));

    size_t srcImageSize = image.channels() * sizeof(uint8_t) * image.rows * image.cols;
    cl::Buffer srcImageBuffer(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        srcImageSize,
        image.data,
        &status
    );
    verifyOpenCL_Status(status);

    size_t scaledImageSize = scaledImage.channels() * sizeof(uint8_t) * scaledImage.rows * scaledImage.cols;
    cl::Buffer scaledImageBuffer(
        mContext,
        CL_MEM_WRITE_ONLY,
        scaledImageSize,
        nullptr,
        &status
    );
    verifyOpenCL_Status(status);

    verifyOpenCL_Status(mKernel.setArg(0, srcImageBuffer));
    verifyOpenCL_Status(mKernel.setArg(1, cl_uint(image.step1())));
    verifyOpenCL_Status(mKernel.setArg(2, scaledImageBuffer));
    verifyOpenCL_Status(mKernel.setArg(3, cl_uint(scaledImage.step1())));
    verifyOpenCL_Status(mKernel.setArg(4, cl_uint(image.channels())));
    verifyOpenCL_Status(mKernel.setArg(5, cl_float(scaleX)));
    verifyOpenCL_Status(mKernel.setArg(6, cl_float(scaleY)));

    verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(scaledImage.rows, scaledImage.cols), cl::NDRange(1, 1)));
    verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(scaledImageBuffer, CL_TRUE, 0, scaledImageSize, scaledImage.data));

    return scaledImage;
}

std::string svp::NNI_OpenCL::toString() {
    return "Scaling image using Nearest Neighbour Interpolation using OpenCL";
}

void svp::NNI_OpenCL2::init() {
    if(isInitialised) return;

    cl_int status = CL_SUCCESS;

    mContext = cl::Context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    svp::verifyOpenCL_Status(status);

    auto devices = mContext.getInfo<CL_CONTEXT_DEVICES>(&status);
    svp::verifyOpenCL_Status(status);
    mDevice = devices.front();

    cl::Program program(
        mContext,
        svp::readScript("ImageScaling.cl"),
        false,
        &status
    );
    svp::verifyOpenCL_Status(status);
    svp::verifyOpenCL_Status(program.build("-cl-std=CL1.2"));
    mKernel = cl::Kernel(program, "nearestNeighbourInterpolation2", &status);
    svp::verifyOpenCL_Status(status);

    mCommandQueue = cl::CommandQueue(mContext, mDevice, 0, &status);
    svp::verifyOpenCL_Status(status);

    isInitialised = true;
}

svp::NNI_OpenCL2::NNI_OpenCL2() {
    init();
}

cv::Mat svp::NNI_OpenCL2::transform(const cv::Mat &image, float scaleX, float scaleY) {
    SVP_PROFILE_FUNC();

    cl_int status;
    cv::Mat scaledImage(std::round(image.rows * scaleY), std::round(image.cols * scaleX), CV_8UC(image.channels()));

    cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);

    cl::Image2D srcImage2D(
        mContext,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        imageFormat,
        image.cols,
        image.rows,
        0,
        image.data,
        &status
    );
    verifyOpenCL_Status(status);

    cl::Image2D scaledImage2D(
        mContext,
        CL_MEM_WRITE_ONLY,
        imageFormat,
        scaledImage.cols,
        scaledImage.rows,
        0,
        nullptr,
        &status
    );
    verifyOpenCL_Status(status);

    verifyOpenCL_Status(mKernel.setArg(0, srcImage2D));
    verifyOpenCL_Status(mKernel.setArg(1, scaledImage2D));
    verifyOpenCL_Status(mKernel.setArg(2, cl_float(scaleX)));
    verifyOpenCL_Status(mKernel.setArg(3, cl_float(scaleY)));

    verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
        mKernel,
        cl::NullRange,
        cl::NDRange(scaledImage.rows, scaledImage.cols),
        cl::NDRange(1, 1)
    ));

    cl::size_t<3> region;
    region[0] = scaledImage.cols, region[1] = scaledImage.rows, region[2] = 1;
    verifyOpenCL_Status(mCommandQueue.enqueueReadImage(
        scaledImage2D,
        CL_TRUE,
        cl::size_t<3>(),
        region,
        scaledImage.step1(),
        0,
        scaledImage.data
    ));

    return scaledImage;
}

std::string svp::NNI_OpenCL2::toString() {
    return "Scaling image using Nearest Neighbour Interpolation using OpenCL with inbuilt image functionality";
}
