#include "MatrixMultiplication.h"

int main(int argc, char **argv) {
    svp::DotProductBenchMarker dotProductBenchMarker;

    size_t size = 1024;
    auto matrix1 = svp::Matrix(size, std::vector<svp::MatrixElementType>(size/2, 1));
    auto matrix2 = svp::Matrix(size/2, std::vector<svp::MatrixElementType>(size, 2));
    auto result = svp::Matrix(size, std::vector<svp::MatrixElementType>(size, size));

    for(auto &pDotProductStrategy: {
        static_cast<svp::DotProductStrategy*>(new svp::SerialDotProductStrategy()),
        static_cast<svp::DotProductStrategy*>(new svp::OpenMP_DotProductStrategy()),
        static_cast<svp::DotProductStrategy*>(new svp::OpenCL_DotProductStrategy()),
    }) {
        dotProductBenchMarker.setDotProductStrategy(std::unique_ptr<svp::DotProductStrategy>(pDotProductStrategy));
        dotProductBenchMarker.benchmarkCalculateDotProduct(
            10,
            matrix1,
            matrix2,
            result
        );
    }

    return 0;
}
