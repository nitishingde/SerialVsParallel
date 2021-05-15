#include <cmath>
#include "Pi.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    const int32_t ITERATIONS = 10;
    for(auto &pPiStrategy: {
        std::shared_ptr<svp::PiStrategy>(new svp::SerialPiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::OpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::CacheFriendlyOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::AtomicBarrierOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::ReductionOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::OpenCL_PiStrategy()),
    }) {
        double pi = 0;
        SVP_START_BENCHMARKING_SESSION(pPiStrategy->toString().c_str(), ITERATIONS) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            pi += pPiStrategy->calculatePi(1e8);
        }

        pi /= ITERATIONS;
        printf("Pi                : %0.17g\n", pi);
        printf("Error margin      : %0.17g %%\n", (std::abs(pi-M_PI)*100.0)/M_PI);
    }

    return 0;
}
