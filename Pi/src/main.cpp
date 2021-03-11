#include "Pi.h"

int main(int argc, char **argv) {
    PiBenchMarker piBenchMarker;

    for(auto &pPiStrategy: {
        static_cast<PiStrategy*>(new SerialPiStrategy()),
    }) {
        piBenchMarker.setPiStrategy(std::unique_ptr<PiStrategy>(pPiStrategy));
        piBenchMarker.benchmarkCalculatePi(10, 1e8);
    }

    return 0;
}
