#include "Pi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

double SerialPiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    double area = 0.0;

    for(uint32_t step = 0; step < steps; ++step) {
        double x = (step + 0.5) * delta;
        area += 4.0 / (1.0 + x*x);
    }

    return area * delta;
}

std::string SerialPiStrategy::toString() {
    return "Calculate Pi using serial code";
}

PiBenchMarker::PiBenchMarker(std::unique_ptr<PiStrategy> pPiStrategy)
    : mpPiStrategy(std::move(pPiStrategy))
{}

void PiBenchMarker::setPiStrategy(std::unique_ptr<PiStrategy> pPiStrategy) {
    mpPiStrategy = std::move(pPiStrategy);
}

void PiBenchMarker::benchmarkCalculatePi(uint32_t iterations, uint32_t steps) const {
    std::vector<double> executionTime(iterations, 0.0);
    double pi = 0.0;

    for(uint32_t iteration = 0; iteration < executionTime.size(); ++iteration) {
        printf("\rIteration: %u/%u", iteration+1, iterations);
        fflush(stdout);

        auto start = std::chrono::high_resolution_clock::now();
        pi = mpPiStrategy->calculatePi(steps);
        auto end = std::chrono::high_resolution_clock::now();
        executionTime[iteration] = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1.e9;
    }

    printf("\r");
    printf("> Strategy        : %s\n", mpPiStrategy->toString().c_str());
    printf("> Iterations      : %u\n", iterations);
    printf("> Steps           : %u\n", steps);
    printf("Pi                : %0.17g\n", pi);
    printf("Avg Execution Time: %.9gs\n", std::accumulate(executionTime.begin(), executionTime.end(), 0.0)/executionTime.size());
    printf("Min Execution Time: %.9gs\n", *std::min_element(executionTime.begin(), executionTime.end()));
    printf("Max Execution Time: %.9gs\n", *std::max_element(executionTime.begin(), executionTime.end()));
    printf("\n");
}
