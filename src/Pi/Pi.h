#ifndef SERIALVSPARALLEL_PI_H
#define SERIALVSPARALLEL_PI_H


#include <memory>
#include <CL/cl.hpp>
#include "../Utility.h"

namespace svp {
    class PiStrategy {
    public:
        virtual ~PiStrategy() = default;
        virtual double calculatePi(uint32_t steps) = 0;
        virtual std::string toString() = 0;
    };

    class SerialPiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class OpenMP_PiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class CacheFriendlyOpenMP_PiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class AtomicBarrierOpenMP_PiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class ReductionOpenMP_PiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class OpenCL_PiStrategy : public PiStrategy, private OpenCL_Base {
    private:
        cl::Kernel mKernel;

    private:
        void init() override;
    public:
        explicit OpenCL_PiStrategy();
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class MPI_PiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };

    class HybridMpiOpenMP_PiStrategy : public PiStrategy {
    public:
        double calculatePi(uint32_t steps) override;
        std::string toString() override;
    };
}


#endif //SERIALVSPARALLEL_PI_H
