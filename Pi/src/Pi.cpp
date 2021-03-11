#include "Pi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <omp.h>
#include <vector>

#define CACHE_PADDING 8

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

double OpenMP_PiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    std::vector<double> area(omp_get_max_threads()*2, 0.0);

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel firstprivate(steps, delta) shared(area) default(none)
    {
        auto totalThreads = omp_get_num_threads();
        auto threadID = omp_get_thread_num();
        for(uint32_t step = threadID; step < steps; step+=totalThreads) {
            double x = (step + 0.5) * delta;
            area[threadID] += 4.0 / (1.0 + x*x);
        }
    }

    return std::accumulate(area.begin(), area.end(), 0.0) * delta;
}

std::string OpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, simplified";
}

double CacheFriendlyOpenMP_PiStrategy::calculatePi(uint32_t steps) {
    const double delta = 1.0 / steps;
    uint32_t maxThreadsPossible = omp_get_max_threads();
    double area[maxThreadsPossible][CACHE_PADDING];
    for(uint32_t i = 0; i < maxThreadsPossible; ++i) {
        area[i][0] = 0;
    }

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel firstprivate(steps, delta) shared(area) default(none)
    {
        auto totalThreads = omp_get_num_threads();
        auto threadID = omp_get_thread_num();
        for(uint32_t step = threadID; step < steps; step+=totalThreads) {
            double x = (step + 0.5) * delta;
            area[threadID][0] += 4.0 / (1.0 + x*x);
        }
    }

    double totalArea = 0.0;
    for(uint32_t i = 0; i < maxThreadsPossible; ++i) {
        totalArea += area[i][0];
    }

    return totalArea * delta;
}

std::string CacheFriendlyOpenMP_PiStrategy::toString() {
    return "Calculate Pi using OpenMP, using cache friendly options";
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
