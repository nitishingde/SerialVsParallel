#ifndef SERIALVSPARALLEL_PRIME_H
#define SERIALVSPARALLEL_PRIME_H


#include "../Utility.h"

namespace svp {
    struct PrimeMetaData {
        uint32_t limit = 2;
        uint32_t largestPrime = 2;
        uint32_t primeCount = 1;
    };

    class PrimeMetaDataStrategy {
    public:
        virtual ~PrimeMetaDataStrategy() = default;
        virtual PrimeMetaData calculateBiggestPrime(const uint32_t limit) = 0;
        virtual std::string toString() = 0;
    };

    class SerialPrimeMetaDataStrategy: public PrimeMetaDataStrategy {
    public:
        PrimeMetaData calculateBiggestPrime(const uint32_t limit) override;
        std::string toString() override;
    };

    class OpenMP_PrimeMetaDataStrategy: public PrimeMetaDataStrategy {
    public:
        PrimeMetaData calculateBiggestPrime(const uint32_t limit) override;
        std::string toString() override;
    };

    class MPI_PrimeMetaDataStrategy: public PrimeMetaDataStrategy {
    public:
        PrimeMetaData calculateBiggestPrime(const uint32_t limit) override;
        std::string toString() override;
    };

    class HybridMpiOpenMP_PrimeMetaDataStrategy: public PrimeMetaDataStrategy {
    public:
        PrimeMetaData calculateBiggestPrime(const uint32_t limit) override;
        std::string toString() override;
    };
}


#endif //SERIALVSPARALLEL_PRIME_H
