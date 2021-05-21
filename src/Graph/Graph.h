#ifndef SERIALVSPARALLEL_GRAPH_H
#define SERIALVSPARALLEL_GRAPH_H


#include <cstdint>
#include <string>
#include <vector>

namespace svp {
    class BfsStrategy {
    public:
        virtual ~BfsStrategy() = default;
        virtual std::vector<int32_t> search(const std::vector<std::vector<int32_t>> &graph, int32_t sourceNode) = 0;
        virtual std::string toString() = 0;
    };

    class SerialBfsStrategy: public BfsStrategy {
    public:
        std::vector<int32_t> search(const std::vector<std::vector<int32_t>> &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class OpenMP_BfsStrategy: public BfsStrategy {
    public:
        std::vector<int32_t> search(const std::vector<std::vector<int32_t>> &graph, int32_t sourceNode) override;
        std::string toString() override;
    };
}

#endif //SERIALVSPARALLEL_GRAPH_H
