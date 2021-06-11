#include <algorithm>
#include <cmath>
#include "Pi.h"

void benchMark(const std::initializer_list<std::shared_ptr<svp::PiStrategy>> &strategies) {
    const int32_t ITERATIONS = 10;

    for(auto &pPiStrategy: strategies) {
        double pi = 0;
        SVP_START_BENCHMARKING_ITERATIONS(ITERATIONS) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            pi += pPiStrategy->calculatePi(1e8);
        }

        pi /= ITERATIONS;
        printf(
            "[Debug] Pi = " GREEN("%17.17f") ", Calculated = " BLUE("%17.17f") ", Error margin = " RED("%g") "%%, Strategy = " YELLOW("%s\n"),
            M_PI,
            pi,
            (100*std::abs(pi-M_PI))/M_PI,
            pPiStrategy->toString().c_str()
        );
    }
}

int main(int argc, char **argv) {
    SVP_START_BENCHMARKING_SESSION("Calculate Pi");
    std::vector<std::string> arguments(argv, argv + argc);

    if(std::find(arguments.begin(), arguments.end(), "--use-mpi") != arguments.end()) {
        svp::MPI_GlobalLockGuard lock(argc, argv);
        benchMark({
            std::shared_ptr<svp::PiStrategy>(new svp::MPI_PiStrategy()),
            std::shared_ptr<svp::PiStrategy>(new svp::HybridMpiOpenMP_PiStrategy()),
        });
        // the work of non root processes is completed here
        if(!svp::isMpiRootPid()) return 0;
    }

    // only root process executes this
    benchMark({
        std::shared_ptr<svp::PiStrategy>(new svp::SerialPiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::OpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::CacheFriendlyOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::AtomicBarrierOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::ReductionOpenMP_PiStrategy()),
        std::shared_ptr<svp::PiStrategy>(new svp::OpenCL_PiStrategy()),
    });

    return 0;
}
