#ifndef SERIALVSPARALLEL_MATRIXMULTIPLICATION_H
#define SERIALVSPARALLEL_MATRIXMULTIPLICATION_H


#include <CL/cl.hpp>
#include <memory>
#include <string>
#include <vector>

namespace svp {
    using Matrix = std::vector<std::vector<float>>;
    using MatrixElementType = Matrix::value_type::value_type;

    class DotProductStrategy {
    protected:
        static bool verifyMatrices(const Matrix &matrix1, const Matrix &matrix2, const Matrix &result);
    public:
        virtual ~DotProductStrategy() = default;
        virtual void calculateDotProduct(const Matrix &matrix1, const Matrix &matrix2, Matrix &result) = 0;
        virtual std::string toString() = 0;
    };

    class SerialDotProductStrategy: public DotProductStrategy {
    public:
        void calculateDotProduct(const Matrix &matrix1, const Matrix &matrix2, Matrix &result) override;
        std::string toString() override;
    };

    class OpenMP_DotProductStrategy: public DotProductStrategy {
    public:
        void calculateDotProduct(const Matrix &matrix1, const Matrix &matrix2, Matrix &result) override;
        std::string toString() override;
    };

    class OpenCL_DotProductStrategy: public DotProductStrategy {
    private:
        bool isInitialised = false;
        size_t mWorkGroupSize2D {};
        cl::Context mContext;
        cl::Device mDevice;
        cl::Kernel mKernel;
        cl::CommandQueue mCommandQueue;

    private:
        void init();
    public:
        explicit OpenCL_DotProductStrategy();
        void calculateDotProduct(const Matrix &matrix1, const Matrix &matrix2, Matrix &result) override;
        std::string toString() override;
    };
}


#endif //SERIALVSPARALLEL_MATRIXMULTIPLICATION_H
