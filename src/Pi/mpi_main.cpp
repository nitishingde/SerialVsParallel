#include "Pi.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    svp::MPI_GlobalLockGuard lock(argc, argv);

    svp::PiBenchMarker piBenchMarker;
    for(auto &pPiStrategy: {
        static_cast<svp::PiStrategy*>(new svp::MPI_PiStrategy()),
        static_cast<svp::PiStrategy*>(new svp::HybridMpiOpenMP_PiStrategy()),
    }) {
        piBenchMarker.setPiStrategy(std::unique_ptr<svp::PiStrategy>(pPiStrategy));
        piBenchMarker.benchmarkCalculatePi(10, 1e8);
    }

    return 0;
}
