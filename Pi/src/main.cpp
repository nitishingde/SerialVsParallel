#include "Pi.h"

int main(int argc, char **argv) {
    PiBenchMarker piBenchMarker;

    piBenchMarker.setPiStrategy(std::make_unique<SerialPiStrategy>());
    piBenchMarker.benchmarkCalculatePi(10, 1e6);
    piBenchMarker.benchmarkCalculatePi(10, 1e7);
    piBenchMarker.benchmarkCalculatePi(10, 1e8);

    return 0;
}
