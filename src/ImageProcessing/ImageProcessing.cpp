#include "ImageProcessing.h"
#include <chrono>
#include <numeric>

cv::Mat svp::NNI_Serial::transform(const cv::Mat &image, float scaleX, float scaleY) {
    auto channelSize = image.channels();
    cv::Mat scaledImage(std::round(image.rows * scaleY), std::round(image.cols * scaleX), CV_8UC(channelSize));
    for(size_t scaledI = 0; scaledI < scaledImage.rows; ++scaledI) {
        for(size_t scaledJ = 0; scaledJ < scaledImage.cols; ++scaledJ) {
            size_t scaledImageOffset = scaledI*scaledImage.step1() + scaledJ*channelSize;

            size_t i = scaledI / scaleY, j = scaledJ / scaleX;
            size_t imageOffset = i*image.step1() + j*channelSize;

            for(size_t channel = 0; channel < channelSize; ++channel) {
                scaledImage.data[scaledImageOffset + channel] = image.data[imageOffset + channel];
            }
        }
    }

    return scaledImage;
}

std::string svp::NNI_Serial::toString() {
    return "Scaling image using Nearest Neighbour Interpolation using serial code";
}

svp::ImageScalingBenchMarker::ImageScalingBenchMarker(std::unique_ptr<ImageScalingStrategy> pImageScalingStrategy)
    : mpImageScalingStrategy(std::move(pImageScalingStrategy)) {
}

void svp::ImageScalingBenchMarker::setImageScalingStrategy(std::unique_ptr<ImageScalingStrategy> pImageScalingStrategy) {
    mpImageScalingStrategy = std::move(pImageScalingStrategy);
}

void svp::ImageScalingBenchMarker::benchmarkImageScaling(uint32_t iterations, const cv::Mat &image, float scaleX, float scaleY, const cv::Mat &expectedResult) const {
    std::vector<double> executionTime(iterations, 0.0);

    for(uint32_t iteration = 0; iteration < executionTime.size(); ++iteration) {
#if NDEBUG
        printf("\rIteration: %u/%u", iteration+1, iterations);
        fflush(stdout);
#endif

        auto start = std::chrono::high_resolution_clock::now();
        auto scaledImage = mpImageScalingStrategy->transform(image, scaleX, scaleY);
        auto end = std::chrono::high_resolution_clock::now();
        executionTime[iteration] = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/1.e9;
#if not NDEBUG
        if(!std::equal(expectedResult.begin<uint8_t>(), expectedResult.end<uint8_t>(), scaledImage.begin<uint8_t>())) {
            return;
        }
        printf("[Debug] Execution Time for iteration (%u, %u): %0.9gs\n", iteration+1, iterations, executionTime[iteration]);
#endif
    }

    printf("\r");
    printf("> Strategy        : %s\n", mpImageScalingStrategy->toString().c_str());
    printf("> Iterations      : %u\n", iterations);
    printf("> ~Loops/iteration: %g\n", double(expectedResult.total())*double(expectedResult.channels()));
    printf("Avg Execution Time: %.9gs\n", std::accumulate(executionTime.begin(), executionTime.end(), 0.0)/executionTime.size());
    printf("Min Execution Time: %.9gs\n", *std::min_element(executionTime.begin(), executionTime.end()));
    printf("Max Execution Time: %.9gs\n", *std::max_element(executionTime.begin(), executionTime.end()));
    printf("\n");
}
