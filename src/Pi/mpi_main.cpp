#include "Pi.h"
#include "../Utility.h"

int main(int argc, char **argv) {
    MPI_GlobalLockGuard lock(argc, argv);

    PiBenchMarker piBenchMarker(std::make_unique<MPI_PiStrategy>());
    piBenchMarker.benchmarkCalculatePi(10, 1e8);

    return 0;
}
