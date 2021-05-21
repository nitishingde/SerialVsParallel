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

std::vector<int32_t> readAnswer(const char *pFileName) {
    std::ifstream input(pFileName);
    std::vector<int32_t> distances;
    for(int32_t distance;!input.eof();) {
        input >> distance;
        distances.emplace_back(distance);
    }
    return distances;
}

int main(int argc, char **argv) {
    auto graph = readsAdjacencyList("resources/hugetrace-00010.mtx.adj");
    auto sourceNode = 0;
#if not NDEBUG
    auto check = readAnswer("resources/hugetrace-00010.mtx.ans0");
#endif

    for(const auto &pStrategy: {
        std::shared_ptr<svp::BfsStrategy>(new svp::SerialBfsStrategy()),
    }) {
        std::vector<int32_t> distance;
        SVP_START_BENCHMARKING_SESSION(pStrategy->toString().c_str(), 10) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            distance = pStrategy->search(graph, sourceNode);
        }
#if not NDEBUG
        if(check != distance) {
            printf("[Debug] Failed!\n");
            return -1;
        }
#else
        if(distance.empty() or distance.size() < sourceNode or distance[sourceNode] != 0) {
            printf("[Debug] Failed!\n");
            return -1;
        }
#endif
    }

    return 0;
}