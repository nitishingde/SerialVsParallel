#include "Pi.h"

int main(int argc, char **argv) {
    PiBenchMarker piBenchMarker(std::make_unique<MPI_PiStrategy>());
    piBenchMarker.benchmarkCalculatePi(1, 1e8);

    return 0;
}
