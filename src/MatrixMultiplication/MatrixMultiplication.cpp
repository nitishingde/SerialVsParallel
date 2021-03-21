#include "MatrixMultiplication.h"
#include <algorithm>
#include <chrono>
#include <numeric>

bool svp::DotProductStrategy::verifyMatrices(const svp::Matrix &matrix1, const svp::Matrix &matrix2, const svp::Matrix &result) {
    return
        matrix1.size() == result.size()
        and matrix2[0].size() == result[0].size()
        and matrix1[0].size() == matrix2.size()
        ;
}

void svp::SerialDotProductStrategy::calculateDotProduct(const svp::Matrix &matrix1, const svp::Matrix &matrix2, svp::Matrix &result) {
    if(!verifyMatrices(matrix1, matrix2, result)) return;

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

svp::DotProductBenchMarker::DotProductBenchMarker(std::unique_ptr<DotProductStrategy> pDotProductStrategy)
    : mpDotProductStrategy(std::move(pDotProductStrategy)) {}

void svp::DotProductBenchMarker::setDotProductStrategy(std::unique_ptr<DotProductStrategy> pDotProductStrategy) {
    mpDotProductStrategy = std::move(pDotProductStrategy);
}

void svp::DotProductBenchMarker::benchmarkCalculateDotProduct(uint32_t iterations, const svp::Matrix &matrix1, const svp::Matrix &matrix2, const svp::Matrix &expectedResult) const {
    std::vector<double> executionTime(iterations, 0.0);

    auto calculatedMatrix = Matrix(expectedResult.size(), std::vector<svp::MatrixElementType>(expectedResult[0].size(), -1));
    for(uint32_t iteration = 0; iteration < executionTime.size(); ++iteration) {
#if NDEBUG
        printf("\rIteration: %u/%u", iteration+1, iterations);
        fflush(stdout);
#endif

        auto start = std::chrono::high_resolution_clock::now();
        mpDotProductStrategy->calculateDotProduct(matrix1, matrix2, calculatedMatrix);
        auto end = std::chrono::high_resolution_clock::now();
        executionTime[iteration] = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1.e9;
#if not NDEBUG
        for(size_t i = 0; i < expectedResult.size(); ++i) {
            for(size_t j = 0; j < expectedResult[i].size(); ++j) {
                if(expectedResult[i][j] != calculatedMatrix[i][j]) {
                    printf("[Debug] Expected, calculated = (%f, %f) @ (%zu, %zu)\n", expectedResult[i][j], calculatedMatrix[i][j], i, j);
                    return;
                }
            }
        }
        printf("[Debug] Execution Time for iteration (%u, %u): %0.9gs\n", iteration+1, iterations, executionTime[iteration]);
#endif
    }

    printf("\r");
    printf("> Strategy        : %s\n", mpDotProductStrategy->toString().c_str());
    printf("> Iterations      : %u\n", iterations);
    printf("> ~Loops/iteration: %g\n", (double)matrix1.size()*(double)matrix2.size()*(double)matrix2[0].size());
    printf("Avg Execution Time: %.9gs\n", std::accumulate(executionTime.begin(), executionTime.end(), 0.0)/executionTime.size());
    printf("Min Execution Time: %.9gs\n", *std::min_element(executionTime.begin(), executionTime.end()));
    printf("Max Execution Time: %.9gs\n", *std::max_element(executionTime.begin(), executionTime.end()));
    printf("\n");
}
