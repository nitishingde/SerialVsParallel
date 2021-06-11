#include <algorithm>
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

void benchMark(const std::initializer_list<std::shared_ptr<svp::PrimeMetaDataStrategy>> &strategies, const std::vector<uint32_t> &primes) {
    const uint32_t LIMIT = 1e7;

    for(auto &pStrategy: strategies) {
        SVP_START_BENCHMARKING_ITERATIONS(10) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto op = pStrategy->calculateBiggestPrime(LIMIT);
            // verify
            for(uint32_t i = primes.size()-1; 0 <= i; --i) {
                if(primes[i] <= LIMIT) {
                    if(op.largestPrime != primes[i]) {
                        setlocale(LC_NUMERIC, "");
                        printf("[Debug] Largest Prime: " "Expected = " GREEN("%'u") ", Calculated = " RED("%'u\n"), primes[i], op.largestPrime);
                    }
                    if(op.primeCount != (i+1)) {
                        setlocale(LC_NUMERIC, "");
                        printf("[Debug] Prime Count  : " "Expected = " GREEN("%'u") ", Calculated = " RED("%'u\n"), i+1, op.primeCount);
                    }
                    break;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    SVP_START_BENCHMARKING_SESSION("Prime");
    std::vector<uint32_t> primes;
    if(svp::isMpiRootPid()) {
        primes = get1stMillionPrimes();
    }

    std::vector<std::string> arguments(argv, argv + argc);
    if(std::find(arguments.begin(), arguments.end(), "--use-mpi") != arguments.end()) {
        svp::MPI_GlobalLockGuard lock(argc, argv);
        benchMark(
            {
                std::shared_ptr<svp::PrimeMetaDataStrategy>(new svp::MPI_PrimeMetaDataStrategy()),
                std::shared_ptr<svp::PrimeMetaDataStrategy>(new svp::HybridMpiOpenMP_PrimeMetaDataStrategy()),
            },
            primes
        );
        // the work of non root processes is completed here
        if(!svp::isMpiRootPid()) return 0;
    }

    // only root process executes this
    benchMark(
        {
            std::shared_ptr<svp::PrimeMetaDataStrategy>(new svp::SerialPrimeMetaDataStrategy()),
            std::shared_ptr<svp::PrimeMetaDataStrategy>(new svp::OpenMP_PrimeMetaDataStrategy()),
        },
        primes
    );

    return 0;
}
