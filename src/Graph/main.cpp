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

        std::string weight;
        for(int32_t i = 0; i < edgeCount; ++i) {
            input >> weight;
            weightList.emplace_back(std::stof(weight));
        }
    }

    return graph;
}

template<typename T = int32_t>
std::vector<T> readAnswer(const char *pFileName) {
#if DEBUG_GRAPH
    return {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
#endif
    std::ifstream input(pFileName);
    std::vector<T> distances;
    for(T distance;!input.eof();) {
        if(std::is_same_v<T, float>) {
            std::string no;
            input >> no;
            distance = std::stof(no);
        } else {
            input >> distance;
        }
        distances.emplace_back(distance);
    }

    return distances;
}

void benchMarkBfs(const svp::CsrGraph &graph, const int32_t sourceNode, const std::vector<int32_t> &check) {
#if not NDEBUG
    auto iterations = 1;
#else
    auto iterations = 10;
#endif

    SVP_START_BENCHMARKING_SESSION("BFS");
    for(const auto &pStrategy: {
        std::shared_ptr<svp::BfsStrategy>(new svp::SerialBfsStrategy()),
        std::shared_ptr<svp::BfsStrategy>(new svp::OpenMP_BfsStrategy()),
        std::shared_ptr<svp::BfsStrategy>(new svp::OpenCL_BfsStrategy()),
    }) {
        SVP_START_BENCHMARKING_ITERATIONS(iterations) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto result = pStrategy->search(graph, sourceNode);
            if(!svp::verifyLineage(graph, result.parents, result.rootNode)) {
                fprintf(stderr, "[Debug] Failed, wrong lineage\n");
                return;
            }
            const auto &costs = result.costs;
            for(uint32_t i = 0; i < costs.size(); ++i) {
                if(check[i] != costs[i]) {
                    fprintf(stderr, "[Debug] Failed! size = %zu, node = %u, expected = %d, calculated = %d\n", check.size(), i, check[i], costs[i]);
                    return;
                }
            }
        }
    }
}

void benchMarkDijkstra(const svp::CsrGraph &graph, const int32_t sourceNode, const std::vector<float> &check) {
#if not NDEBUG
    auto iterations = 1;
#else
    auto iterations = 10;
#endif

    SVP_START_BENCHMARKING_SESSION("Dijkstra");
    for(const auto &pStrategy: {
        std::shared_ptr<svp::DijkstraStrategy>(new svp::SerialDijkstraStrategy()),
        std::shared_ptr<svp::DijkstraStrategy>(new svp::OpenCL_DijkstraStrategy()),
    }) {
        SVP_START_BENCHMARKING_ITERATIONS(iterations) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto result = pStrategy->calculate(graph, sourceNode);
            if(!svp::verifyLineage(graph, result.parents, result.rootNode)) {
                fprintf(stderr, "[Debug] Failed, wrong lineage\n");
                return;
            }
            const auto &costs = result.costs;
            for(uint32_t i = 0; i < costs.size(); ++i) {
                if(0.001f < std::abs(check[i]-costs[i])) {
                    fprintf(stderr, "[Debug] Failed! size = %zu, node = %u, expected = %f, calculated = %f\n", check.size(), i, check[i], costs[i]);
                    return;
                }
            }
        }
    }
}

void benchmarkFloydWarshall(const svp::CsrGraph &graph, const std::vector<std::vector<float>> &check) {
#if not NDEBUG
    auto iterations = 1;
#else
    auto iterations = 10;
#endif

    SVP_START_BENCHMARKING_SESSION("Floyd Warshall");
    for(const auto &pStrategy: {
        std::shared_ptr<svp::FloydWarshallStrategy>(new svp::SerialFloydWarshallStrategy()),
        std::shared_ptr<svp::FloydWarshallStrategy>(new svp::OpenMP_FloydWarshallStrategy()),
    }) {
        SVP_START_BENCHMARKING_ITERATIONS(iterations) {
            SVP_PRINT_BENCHMARKING_ITERATION();
            auto result = pStrategy->calculate(graph);
            for(uint32_t i = 0; i < result.size(); ++i) {
                for(uint32_t j = 0; j < result[i].size(); ++j) {
                    if(0.001 < std::abs(result[i][j] - check[i][j])) {
                        fprintf(stderr, "[Debug] Failed! size = %zu, node = %u, expected = %f, calculated = %f\n", check.size(), i, check[i][j], result[i][j]);
                        return;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    for(auto &file: std::vector<std::string> {
        "resources/gre_512.mtx",
        "resources/cage8.mtx",
        "resources/appu.mtx",
        "resources/kron_g500-logn16.mtx",
        "resources/cage13.mtx",
        "resources/ca2010.mtx",
    }) {
        printf("Graph: " GREEN("%s\n"), file.c_str());
        auto graph = getCsrGraph((file+".adj").c_str());
        setlocale(LC_NUMERIC, "");
        printf("Nodes: " BLUE("%'zu") " Edges: " RED("%'zu\n"), graph.getVertexCount(), graph.edgeList.size());
        auto sourceNode = 0;

        {
            auto check = readAnswer((file+".ans").c_str());
            benchMarkBfs(graph, sourceNode, check);
        }


        {
            auto check = readAnswer<float>((file + ".ans.sssp").c_str());
            benchMarkDijkstra(graph, sourceNode, check);
        }

        if(graph.getVertexCount() <= 1024) {
            svp::SerialDijkstraStrategy dijkstraStrategy;
            std::vector<std::vector<float>> check;
            for(int32_t node = 0; node < graph.getVertexCount(); ++node) {
                auto tree = dijkstraStrategy.calculate(graph, node);
                check.emplace_back(tree.costs);
            }
            benchmarkFloydWarshall(graph, check);
        }

        printf("\n");
    }

    return 0;
}
