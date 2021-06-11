#include "Prime.h"
#include <mpi.h>
#include <omp.h>

svp::PrimeMetaData svp::SerialPrimeMetaDataStrategy::calculateBiggestPrime(const uint32_t limit) {
    SVP_PROFILE_SCOPE(toString().c_str());

    if(limit < 3) {
        return {
            limit,
            2,
            1
        };
    }

    PrimeMetaData primeMetaData{limit, 3, 2};

    for(uint32_t no = 5; no <= limit; no+=2) {
        bool primeFound = true;
        for(uint32_t divisor = 3; divisor*divisor <= no; divisor+=2) {
            if(no%divisor == 0) {
                primeFound = false;
                break;
            }
        }
        if(primeFound) {
            primeMetaData.largestPrime = no;
            primeMetaData.primeCount++;
        }
    }

    return primeMetaData;
}

std::string svp::SerialPrimeMetaDataStrategy::toString() {
    return "Get #prime and the largest prime <= Limit";
}

svp::PrimeMetaData svp::OpenMP_PrimeMetaDataStrategy::calculateBiggestPrime(const uint32_t limit) {
    SVP_PROFILE_SCOPE(toString().c_str());

    if(limit < 3) {
        return {
            limit,
            2,
            1
        };
    }

    PrimeMetaData primeMetaData{limit, 3, 2};
    omp_set_num_threads(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic, 4) default(none) firstprivate(limit) shared(primeMetaData)
    for(uint32_t no = 5; no <= limit; no+=2) {
        bool primeFound = true;
        for(uint32_t divisor = 3; divisor*divisor <= no; divisor+=2) {
            if(no%divisor == 0) {
                primeFound = false;
                break;
            }
        }
        if(primeFound) {
            #pragma omp critical
            {
                primeMetaData.largestPrime = std::max(primeMetaData.largestPrime, no);
                primeMetaData.primeCount++;
            }
        }
    }

    return primeMetaData;
}

std::string svp::OpenMP_PrimeMetaDataStrategy::toString() {
    return "Get #prime and the largest prime <= Limit using OpenMP";
}

svp::PrimeMetaData svp::MPI_PrimeMetaDataStrategy::calculateBiggestPrime(const uint32_t limit) {
    SVP_PROFILE_SCOPE(toString().c_str());

    int32_t processId, noOfProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &noOfProcesses);

    svp::PrimeMetaData primeMetaData {
        limit,
        0,
        0
    };
    uint32_t largestPrime = 0;
    uint32_t primeCount = 0;
    uint32_t start = 2*processId + 1;
    uint32_t stride = 2*noOfProcesses;
    // for the root process, don't start at 1, since 1 is neither prime nor composite
    if(processId == 0) {
        start += stride;
        largestPrime = 2;
        primeCount = 1;
    }

    for(uint32_t no = start; no <= limit; no+=stride) {
        bool primeFound = true;
        for(uint32_t divisor = 3; divisor*divisor <= no; divisor+=2) {
            if(no%divisor == 0) {
                primeFound = false;
                break;
            }
        }
        if(primeFound) {
            largestPrime = no;
            primeCount++;
        }
    }

    MPI_Reduce(&largestPrime, &primeMetaData.largestPrime, 1, MPI_UINT32_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&primeCount, &primeMetaData.primeCount, 1, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

    return primeMetaData;
}

std::string svp::MPI_PrimeMetaDataStrategy::toString() {
    return "Get #prime and the largest prime <= Limit using MPI";
}
