#include <algorithm>
#include <fstream>
#include <memory>
#include "Graph.h"

#define DEBUG_GRAPH 0

svp::CsrGraph getCsrGraph(const char *pFileName) {
#if DEBUG_GRAPH
    return {
        {
            1,2,3,//0
            0,4,5,6,7,//1
            0,7,8,//2
            0,8,9,10,//3
            1,//4
            1,//5
            1,//6
            1,2,//7
            2,3,//8
            3,//9
            3,//10
        },
        {0, 3, 8, 11, 15, 16, 17, 18, 20, 22, 23, 24}
    };
#endif
    std::ifstream input(pFileName);
    std::string type;
    input >> type;
    bool isWeighted = type == "#weighted" ? true: false;

    int32_t N;
    input >> N;

    svp::CsrGraph graph;
    auto &csr = graph.compressedSparseRows;
    csr.resize(N+1);
    csr[0] = 0;
    auto &edgeList = graph.edgeList;
    auto &weightList = graph.weightList;

    for(int32_t n = 0, edgeCount; !input.eof() and n < N; ++n) {
        input >> edgeCount;
        for(int32_t i = 0, vertex; i < edgeCount; ++i) {
            input >> vertex;
            edgeList.emplace_back(vertex);
        }
        csr[n+1] = graph.edgeList.size();

        if(!isWeighted) continue;

        float weight;
        for(int32_t i = 0; i < edgeCount; ++i) {
            input >> weight;
            weightList.emplace_back(weight);
        }
    }

    return graph;
}

std::vector<int32_t> readAnswer(const char *pFileName) {
#if DEBUG_GRAPH
    return {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
#endif
    std::ifstream input(pFileName);
    std::vector<int32_t> distances;
    for(int32_t distance;!input.eof();) {
        input >> distance;
        distances.emplace_back(distance);
    }
    return distances;
}

int main(int argc, char **argv) {
    std::string file =
//        "resources/ca2010.mtx";
        "resources/cage13.mtx";
//        "resources/wiki-Talk.mtx";
    auto graph = getCsrGraph((file+".adj").c_str());
    auto sourceNode = 0;
    auto check = readAnswer((file+".ans"+std::to_string(sourceNode)).c_str());

#if not NDEBUG
    auto iterations = 1;
#else
    auto iterations = 10;
#endif

    for(const auto &pStrategy: {
        std::shared_ptr<svp::BfsStrategy>(new svp::SerialBfsStrategy()),
        std::shared_ptr<svp::BfsStrategy>(new svp::OpenMP_BfsStrategy()),
        std::shared_ptr<svp::BfsStrategy>(new svp::OpenCL_BfsStrategy()),
    }) {
        SVP_START_BENCHMARKING_SESSION(pStrategy->toString().c_str(), iterations) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto distance = pStrategy->search(graph, sourceNode);
            if(check != distance) {
                fprintf(stderr, "[Debug] Failed!\n");
                return -1;
            }
        }
    }

    return 0;
}
