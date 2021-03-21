#ifndef SERIALVSPARALLEL_MATRIXMULTIPLICATION_H
#define SERIALVSPARALLEL_MATRIXMULTIPLICATION_H


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

    class DotProductBenchMarker {
    private:
        std::unique_ptr<DotProductStrategy> mpDotProductStrategy = nullptr;

    public:
        explicit DotProductBenchMarker(std::unique_ptr<DotProductStrategy> pDotProductStrategy = nullptr);
        void setDotProductStrategy(std::unique_ptr<DotProductStrategy> pDotProductStrategy);
        void benchmarkCalculateDotProduct(uint32_t iterations, const Matrix &matrix1, const Matrix &matrix2, const Matrix &expectedResult) const;
    };
}


#endif //SERIALVSPARALLEL_MATRIXMULTIPLICATION_H
