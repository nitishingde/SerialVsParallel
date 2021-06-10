#include "Prime.h"
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
