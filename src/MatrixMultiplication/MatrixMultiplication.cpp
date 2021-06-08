#include "MatrixMultiplication.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <omp.h>
#include "../Utility.h"

bool svp::DotProductStrategy::verifyMatrices(const svp::Matrix &matrix1, const svp::Matrix &matrix2, const svp::Matrix &result) {
    return
        matrix1.size() == result.size()
        and matrix2[0].size() == result[0].size()
        and matrix1[0].size() == matrix2.size()
        ;
}

void svp::SerialDotProductStrategy::calculateDotProduct(const svp::Matrix &matrix1, const svp::Matrix &matrix2, svp::Matrix &result) {
    if(!verifyMatrices(matrix1, matrix2, result)) return;
    SVP_PROFILE_SCOPE(toString().c_str());

    for(size_t i = 0; i < matrix1.size(); ++i) {
        for(size_t j = 0; j < matrix2[0].size(); ++j) {
            result[i][j] = svp::MatrixElementType(0);
            for(size_t k = 0; k < matrix2.size() /*or matrix1[0].size() */; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

std::string svp::SerialDotProductStrategy::toString() {
    return "Calculate Dot Product of 2 matrices using serial code";
}

void svp::OpenMP_DotProductStrategy::calculateDotProduct(const svp::Matrix &matrix1, const svp::Matrix &matrix2, svp::Matrix &result) {
    if(!verifyMatrices(matrix1, matrix2, result)) return;
    SVP_PROFILE_SCOPE(toString().c_str());

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for default(none) shared(matrix1, matrix2, result)
    for(size_t i = 0; i < matrix1.size(); ++i) {
        // Parallelizing inner loops gives pretty bad performance, since threads are spawned and killed multiple no. of times, leading to a large overhead
        //#pragma omp parallel for default(none) firstprivate(i) shared(matrix1, matrix2, result)
        for(size_t j = 0; j < matrix2[0].size(); ++j) {
            result[i][j] = svp::MatrixElementType(0);
            // Apart from being stupidly slow, this has a race condition
            //#pragma omp parallel for default(none) firstprivate(i, j) shared(matrix1, matrix2, result)
            for(size_t k = 0; k < matrix2.size() /*or matrix1[0].size() */; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

std::string svp::OpenMP_DotProductStrategy::toString() {
    return "Calculate Dot Product of 2 matrices using OpenMP";
}

void svp::OpenCL_DotProductStrategy::init() {
    if(isInitialised) return;

    cl_int status = CL_SUCCESS;

    mContext = cl::Context(CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr, &status);
    svp::verifyOpenCL_Status(status);

    auto devices = mContext.getInfo<CL_CONTEXT_DEVICES>(&status);
    svp::verifyOpenCL_Status(status);
    mDevice = devices.front();

    // 512 on my machine
    mWorkGroupSize2D = mDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&status);
    svp::verifyOpenCL_Status(status);
    mWorkGroupSize2D = std::floor(std::sqrt(mWorkGroupSize2D));
#if not NDEBUG
    printf("[DEBUG] Work Group size: %zu\n", mWorkGroupSize2D);
#endif

    cl::Program program(
        mContext,
        svp::readScript("resources/MatrixMultiplication.cl"),
        false,
        &status
    );
    svp::verifyOpenCL_Status(status);
    svp::verifyOpenCL_Status(program.build("-cl-std=CL1.2"));
    mKernel = cl::Kernel(program, "calculateDotProduct", &status);
    svp::verifyOpenCL_Status(status);

    mCommandQueue = cl::CommandQueue(mContext, mDevice, 0, &status);
    svp::verifyOpenCL_Status(status);

    isInitialised = true;
}

svp::OpenCL_DotProductStrategy::OpenCL_DotProductStrategy() {
    init();
}

void svp::OpenCL_DotProductStrategy::calculateDotProduct(const svp::Matrix &matrix1, const svp::Matrix &matrix2, svp::Matrix &result) {
    SVP_PROFILE_SCOPE(toString().c_str());

    cl_int status = CL_SUCCESS;

    cl::Buffer matrix1Buffer(mContext, CL_MEM_READ_ONLY, matrix1.size()*matrix1[0].size()*sizeof(MatrixElementType), nullptr, &status);
    verifyOpenCL_Status(status);
    for(size_t row = 0, rowSize = matrix1[0].size()*sizeof(MatrixElementType); row < matrix1.size(); ++row) {
        verifyOpenCL_Status(mCommandQueue.enqueueWriteBuffer(matrix1Buffer, CL_TRUE, row*rowSize, rowSize, matrix1[row].data()));
    }

    cl::Buffer matrix2Buffer(mContext, CL_MEM_READ_ONLY, matrix2.size()*matrix2[0].size()*sizeof(MatrixElementType), nullptr, &status);
    verifyOpenCL_Status(status);
    for(size_t row = 0, rowSize = matrix2[0].size()*sizeof(MatrixElementType); row < matrix2.size(); ++row) {
        verifyOpenCL_Status(mCommandQueue.enqueueWriteBuffer(matrix2Buffer, CL_TRUE, row*rowSize, rowSize, matrix2[row].data()));
    }

    cl::Buffer resultBuffer(mContext, CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY, result.size()*result[0].size()*sizeof(MatrixElementType), nullptr, &status);
    verifyOpenCL_Status(status);

    verifyOpenCL_Status(mKernel.setArg(0, matrix1Buffer));
    verifyOpenCL_Status(mKernel.setArg(1, matrix2Buffer));
    verifyOpenCL_Status(mKernel.setArg(2, resultBuffer));
    verifyOpenCL_Status(mKernel.setArg(3, static_cast<cl_uint>(matrix1.size())));
    verifyOpenCL_Status(mKernel.setArg(4, static_cast<cl_uint>(matrix2.size())));
    verifyOpenCL_Status(mKernel.setArg(5, static_cast<cl_uint>(matrix2[0].size())));

    size_t globalDimX = std::ceil(result.size()/(double)mWorkGroupSize2D) * mWorkGroupSize2D;
    size_t globalDimY = std::ceil(result[0].size()/(double)mWorkGroupSize2D) * mWorkGroupSize2D;
    verifyOpenCL_Status(mCommandQueue.enqueueNDRangeKernel(
        mKernel,
        cl::NullRange,
        cl::NDRange(globalDimX, globalDimY),
        cl::NDRange(mWorkGroupSize2D, mWorkGroupSize2D),
        nullptr,
        nullptr
    ));

    for(size_t row = 0, rowSize = result[0].size()*sizeof(MatrixElementType); row < result.size(); ++row) {
        verifyOpenCL_Status(mCommandQueue.enqueueReadBuffer(resultBuffer, CL_TRUE, row*rowSize, rowSize, result[row].data(), nullptr));
    }
}

std::string svp::OpenCL_DotProductStrategy::toString() {
    return "Calculate Dot Product of 2 matrices using OpenCL";
}
