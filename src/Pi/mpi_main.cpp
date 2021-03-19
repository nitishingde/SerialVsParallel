#include "Pi.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    MPI_GlobalLockGuard lock(argc, argv);

    PiBenchMarker piBenchMarker;
    for(auto &pPiStrategy: {
        static_cast<PiStrategy*>(new MPI_PiStrategy()),
        static_cast<PiStrategy*>(new HybridMpiOpenMP_PiStrategy()),
    }) {
        piBenchMarker.setPiStrategy(std::unique_ptr<PiStrategy>(pPiStrategy));
        piBenchMarker.benchmarkCalculatePi(10, 1e8);
    }

    return 0;
}
