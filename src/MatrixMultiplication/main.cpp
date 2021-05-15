#include "MatrixMultiplication.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    size_t size = 1024;
    auto matrix1 = svp::Matrix(size, std::vector<svp::MatrixElementType>(size/2, 1));
    auto matrix2 = svp::Matrix(size/2, std::vector<svp::MatrixElementType>(size, 2));
    auto result = svp::Matrix(size, std::vector<svp::MatrixElementType>(size, size));

    for(auto &pDotProductStrategy: {
        std::shared_ptr<svp::DotProductStrategy>(new svp::SerialDotProductStrategy()),
        std::shared_ptr<svp::DotProductStrategy>(new svp::OpenMP_DotProductStrategy()),
        std::shared_ptr<svp::DotProductStrategy>(new svp::OpenCL_DotProductStrategy()),
    }) {
        SVP_START_BENCHMARKING_SESSION(pDotProductStrategy->toString().c_str(), 10) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto calculatedMatrix = svp::Matrix(result.size(), std::vector<svp::MatrixElementType>(result[0].size(), -1));
            pDotProductStrategy->calculateDotProduct(matrix1, matrix2, calculatedMatrix);
#if not NDEBUG
            for(size_t i = 0; i < result.size(); ++i) {
                for(size_t j = 0; j < result[i].size(); ++j) {
                    if(result[i][j] != calculatedMatrix[i][j]) {
                        printf("[Debug] Expected, calculated = (%f, %f) @ (%zu, %zu)\n", result[i][j], calculatedMatrix[i][j], i, j);
                        return -1;
                    }
                }
            }
#endif
        }
    }

    return 0;
}

