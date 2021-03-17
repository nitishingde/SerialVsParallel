#include "Pi.h"

int main(int argc, char **argv) {
    PiBenchMarker piBenchMarker;

    for(auto &pPiStrategy: {
        static_cast<PiStrategy*>(new SerialPiStrategy()),
        static_cast<PiStrategy*>(new OpenMP_PiStrategy()),
        static_cast<PiStrategy*>(new CacheFriendlyOpenMP_PiStrategy()),
        static_cast<PiStrategy*>(new AtomicBarrierOpenMP_PiStrategy()),
        static_cast<PiStrategy*>(new ReductionOpenMP_PiStrategy()),
        static_cast<PiStrategy*>(new OpenCL_PiStrategy()),
    }) {
        piBenchMarker.setPiStrategy(std::unique_ptr<PiStrategy>(pPiStrategy));
        piBenchMarker.benchmarkCalculatePi(10, 1e8);
    }

    return 0;
}
