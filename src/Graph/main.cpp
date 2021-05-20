#include <algorithm>
#include <fstream>
#include <memory>
#include "Graph.h"
#include "../Utility.h"

std::vector<std::vector<int32_t>> readsAdjacencyList(const char *pFileName) {
    int32_t N;
    std::ifstream input(pFileName);
    input >> N;
    input.get();
    std::vector<std::vector<int32_t>> adjacencyList(N);
    for(int32_t n = 0; !input.eof() and n < N; ++n) {
        for(int32_t vertex; input.peek() != '\n';) {
            input >> vertex;
            adjacencyList[n].emplace_back(vertex);
        }
        input.get();
    }

    return adjacencyList;
}

int main(int argc, char **argv) {
    auto graph = readsAdjacencyList("resources/hugetrace-00010.mtx.adj");

    for(const auto &pStrategy: {
        std::shared_ptr<svp::BfsStrategy>(new svp::SerialBfsStrategy()),
    }) {
        SVP_START_BENCHMARKING_SESSION(pStrategy->toString().c_str(), 10) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto distance = pStrategy->search(graph, 0);
        }
    }

    return 0;
}