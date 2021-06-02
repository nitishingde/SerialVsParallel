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

    struct Tree {
        std::vector<float> costs;
        std::vector<int32_t> parents;
    };

    bool verifyLineage(const CsrGraph &graph, const Tree &lineageTree);

    class BfsStrategy {
    public:
        virtual ~BfsStrategy() = default;
        virtual std::vector<int32_t> search(const CsrGraph &graph, int32_t sourceNode) = 0;
        virtual std::string toString() = 0;
    };

    class SerialBfsStrategy: public BfsStrategy {
    public:
        std::vector<int32_t> search(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class OpenMP_BfsStrategy: public BfsStrategy {
    public:
        std::vector<int32_t> search(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class OpenCL_BfsStrategy: public BfsStrategy, private OpenCL_Base {
    private:
        cl::Kernel mKernel;

    private:
        void init() override;
    public:
        OpenCL_BfsStrategy();
        std::vector<int32_t> search(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };

    class DijkstraStrategy {
    public:
        virtual ~DijkstraStrategy() = default;
        virtual Tree calculate(const CsrGraph &graph, int32_t sourceNode) = 0;
        virtual std::string toString() = 0;
    };

    class SerialDijkstraStrategy: public DijkstraStrategy {
    public:
        Tree calculate(const CsrGraph &graph, int32_t sourceNode) override;
        std::string toString() override;
    };
}

#endif //SERIALVSPARALLEL_GRAPH_H
