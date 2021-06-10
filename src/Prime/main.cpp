#include <memory>
#include <fstream>
#include "Prime.h"

std::vector<uint32_t> get1stMillionPrimes() {
    std::ifstream input("resources/primes1e6.txt");
    std::vector<uint32_t> primes(1e6);
    for(uint32_t i = 0; i < 1e6 and !input.eof(); ++i) {
        input >> primes[i];
    }

    return primes;
}

int main() {
    SVP_START_BENCHMARKING_SESSION("Prime");

    const uint32_t LIMIT = 1e7;
    auto primes = get1stMillionPrimes();
    for(auto &pStrategy: {
        std::shared_ptr<svp::PrimeMetaDataStrategy>(new svp::SerialPrimeMetaDataStrategy()),
        std::shared_ptr<svp::PrimeMetaDataStrategy>(new svp::OpenMP_PrimeMetaDataStrategy()),
    }) {
        SVP_START_BENCHMARKING_ITERATIONS(10) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto op = pStrategy->calculateBiggestPrime(LIMIT);
            // verify
            for(uint32_t i = primes.size()-1; 0 <= i; --i) {
                if(primes[i] <= LIMIT) {
                    if(op.largestPrime != primes[i]) {
                        setlocale(LC_NUMERIC, "");
                        printf("[Debug] Largest Prime: " "Expected = " GREEN("%'u") ", Calculated = " RED("%'u\n"), primes[i], op.largestPrime);
                        return 0;
                    }
                    if(op.primeCount != (i+1)) {
                        setlocale(LC_NUMERIC, "");
                        printf("[Debug] Prime Count  : " "Expected = " GREEN("%'u") ", Calculated = " RED("%'u\n"), i+1, op.primeCount);
                        return 0;
                    }
                    break;
                }
            }
        }
    }

    return 0;
}
