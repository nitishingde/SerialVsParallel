#include <cmath>
#include "Pi.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    svp::MPI_GlobalLockGuard lock(argc, argv);
    const int32_t ITERATIONS = 10;

    SVP_START_BENCHMARKING_SESSION("Calculate Pi");
    for(auto &pPiStrategy: {
        std::shared_ptr<svp::PiStrategy>(new svp::MPI_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::HybridMpiOpenMP_PiStrategy()),
    }) {
        double pi = 0;
        SVP_START_BENCHMARKING_ITERATIONS(ITERATIONS) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            pi += pPiStrategy->calculatePi(1e8);
        }

        pi /= ITERATIONS;
        if(0.001 < std::abs(pi-M_PI)) {
            printf("[Debug] Pi = " GREEN("%.9g") ", Calculated = " BLUE("%.9g") ", error margin = " RED("%.9g") "\n", M_PI, pi, pi-M_PI);
        }
    }

    return 0;
}
