#include "Prime.h"

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
