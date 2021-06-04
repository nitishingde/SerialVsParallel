#ifndef SERIALVSPARALLEL_GRAPH_H
#define SERIALVSPARALLEL_GRAPH_H


#include <cstdint>
#include <string>
#include <vector>
#include "../Utility.h"

namespace svp {
    struct CsrGraph {
        std::vector<uint32_t> edgeList;
        std::vector<float> weightList;
        std::vector<uint32_t> compressedSparseRows;
        [[nodiscard]] size_t getVertexCount() const {
            return compressedSparseRows.size()-1;
        }
    };

    template<typename Weight>
    struct WeightedTree {
        explicit WeightedTree(uint32_t nodes, Weight defaultWeight) {
            static_assert(std::is_arithmetic_v<Weight>);
            costs = std::vector<Weight>(nodes, defaultWeight);
            parents = std::vector<int32_t>(nodes, -1);
        }
        std::vector<Weight> costs;
        std::vector<int32_t> parents;
    };

    bool verifyLineage(const CsrGraph &graph, const std::vector<int32_t> &parents);

    class BfsStrategy {
    public:
        virtual ~BfsStrategy() = default;
        virtual WeightedTree<int32_t> search(const CsrGraph &graph, int32_t sourceNode) = 0;
        virtual std::string toString() = 0;
    };

    class SerialBfsStrategy: public BfsStrategy {
    public:
        WeightedTree<int32_t> search(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class OpenMP_BfsStrategy: public BfsStrategy {
    public:
        WeightedTree<int32_t> search(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class OpenCL_BfsStrategy: public BfsStrategy, private OpenCL_Base {
    private:
        cl::Kernel mKernel;

    private:
        void init() override;
    public:
        OpenCL_BfsStrategy();
        WeightedTree<int32_t> search(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class DijkstraStrategy {
    public:
        virtual ~DijkstraStrategy() = default;
        virtual WeightedTree<float> calculate(const CsrGraph &graph, int32_t sourceNode) = 0;
        virtual std::string toString() = 0;
    };

    class SerialDijkstraStrategy: public DijkstraStrategy {
    public:
        WeightedTree<float> calculate(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class OpenCL_DijkstraStrategy: public DijkstraStrategy, private OpenCL_Base {
    private:
        cl::Kernel mCalculateCostKernel;
        cl::Kernel mUpdateCostKernel;

    private:
        void init() override;
    public:
        OpenCL_DijkstraStrategy();
        WeightedTree<float> calculate(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };
}

#endif //SERIALVSPARALLEL_GRAPH_H
