#include "Pi.h"

int main(int argc, char **argv) {
    svp::PiBenchMarker piBenchMarker;

    for(auto &pPiStrategy: {
        static_cast<svp::PiStrategy*>(new svp::SerialPiStrategy()),
        static_cast<svp::PiStrategy*>(new svp::OpenMP_PiStrategy()),
        static_cast<svp::PiStrategy*>(new svp::CacheFriendlyOpenMP_PiStrategy()),
        static_cast<svp::PiStrategy*>(new svp::AtomicBarrierOpenMP_PiStrategy()),
        static_cast<svp::PiStrategy*>(new svp::ReductionOpenMP_PiStrategy()),
        static_cast<svp::PiStrategy*>(new svp::OpenCL_PiStrategy()),
    }) {
        piBenchMarker.setPiStrategy(std::unique_ptr<svp::PiStrategy>(pPiStrategy));
        piBenchMarker.benchmarkCalculatePi(10, 1e8);
    }

    return 0;
}
