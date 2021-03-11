#ifndef SERIALVSPARALLEL_PI_H
#define SERIALVSPARALLEL_PI_H


#include <memory>

class PiStrategy {
public:
    virtual ~PiStrategy() = default;
    virtual double calculatePi(uint32_t steps) = 0;
    virtual std::string toString() = 0;
};

class SerialPiStrategy: public PiStrategy {
public:
    double calculatePi(uint32_t steps) override;
    std::string toString() override;
};

class OpenMP_PiStrategy: public PiStrategy {
public:
    double calculatePi(uint32_t steps) override;
    std::string toString() override;
};

class CacheFriendlyOpenMP_PiStrategy: public PiStrategy {
public:
    double calculatePi(uint32_t steps) override;
    std::string toString() override;
};

class AtomicBarrierOpenMP_PiStrategy: public PiStrategy {
public:
    double calculatePi(uint32_t steps) override;
    std::string toString() override;
};

class PiBenchMarker {
private:
    std::unique_ptr<PiStrategy> mpPiStrategy = nullptr;

public:
    explicit PiBenchMarker(std::unique_ptr<PiStrategy> pPiStrategy = nullptr);
    void setPiStrategy(std::unique_ptr<PiStrategy> pPiStrategy);
    void benchmarkCalculatePi(uint32_t iterations = 10, uint32_t steps = 1000000) const;
};


#endif //SERIALVSPARALLEL_PI_H
