#include <cmath>
#include "Pi.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    const int32_t ITERATIONS = 10;
    SVP_START_BENCHMARKING_SESSION("Calculate Pi");
    for(auto &pPiStrategy: {
        std::shared_ptr<svp::PiStrategy>(new svp::SerialPiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::OpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::CacheFriendlyOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::AtomicBarrierOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::ReductionOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::OpenCL_PiStrategy()),
    }) {
        double pi = 0;
        SVP_START_BENCHMARKING_ITERATIONS(ITERATIONS) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            pi += pPiStrategy->calculatePi(1e8);
        }

        pi /= ITERATIONS;
        if(0.001 < std::abs(pi-M_PI)) {
            printf("[Debug] Pi = %.9g, Calculated = %.9g, error margin = " RED("%.9g"), M_PI, pi, std::abs(pi-M_PI));
            return 0;
        }
    }

    return 0;
}
