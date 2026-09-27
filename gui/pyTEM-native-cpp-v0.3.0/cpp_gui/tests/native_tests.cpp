#include "MatrixConvolution.h"
#include "TemSolver.h"
#include "UsfReader.h"

#include <cmath>
#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void testForwardReference()
{
    pytem::ForwardModel model;
    model.thicknesses = {10.0, 40.0};
    model.resistivities = {100.0, 10.0, 200.0};
    model.times = {1.0e-5, 1.0e-4, 1.0e-3};
    model.txSize = 12.5;
    const auto response = pytem::TemSolver::forward(model);
    const double reference[] = {1.0257663810779942e-4, 1.3133019620386754e-6, 2.542707414147776e-9};
    for (std::size_t i = 0; i < response.size(); ++i) {
        const double relative = std::abs(response[i] - reference[i]) / reference[i];
        require(relative < 2.0e-6, "C++ DLF forward differs from Python reference");
    }
}

void testEulerReference()
{
    pytem::ForwardModel model;
    model.thicknesses = {10.0, 40.0};
    model.resistivities = {100.0, 10.0, 200.0};
    model.times = {1.0e-5, 1.0e-4, 1.0e-3};
    model.txSize = 12.5;
    model.transform = pytem::TransformMethod::Euler;
    model.eulerOrder = 11;
    const auto response = pytem::TemSolver::forward(model);
    const double reference[] = {
        1.02576656161208621e-4,
        1.31330412536200242e-6,
        2.54272929617101088e-9,
    };
    for (std::size_t i = 0; i < response.size(); ++i) {
        const double relative = std::abs(response[i] - reference[i]) / reference[i];
        require(relative < 2.0e-6, "C++ Euler forward differs from Python reference");
    }
}

void testAnalyticalJacobian()
{
    for (const auto transform : {pytem::TransformMethod::DigitalLinearFilter,
                                 pytem::TransformMethod::Euler}) {
        pytem::ForwardModel model;
        model.thicknesses = {10.0, 40.0};
        model.resistivities = {100.0, 10.0, 200.0};
        model.times = {2.0e-5, 2.0e-4};
        model.txSize = 12.5;
        model.rxX = 3.0;
        model.geometry = pytem::Geometry::CircleOffset;
        model.transform = transform;
        model.lowPassFrequencies = {450000.0, 300000.0};
        model.lowPassOrders = {1, 1};
        model.stepTimes = {1.5e-5, 3.0e-5, 1.5e-4, 3.0e-4};
        model.responseMatrix = {{0.4, 0.6, 0.0, 0.0},
                                {0.0, 0.0, 0.3, 0.7}};
        const auto analytical = pytem::TemSolver::logJacobian(
            model, pytem::JacobianMethod::Analytical);
        const auto finiteDifference = pytem::TemSolver::logJacobian(
            model, pytem::JacobianMethod::FiniteDifference, 1.0e-5);
        require(analytical.size() == finiteDifference.size(), "Jacobian row count mismatch");
        for (std::size_t row = 0; row < analytical.size(); ++row) {
            for (std::size_t parameter = 0; parameter < analytical[row].size(); ++parameter) {
                const double scale = std::max(1.0e-8, std::abs(finiteDifference[row][parameter]));
                const double difference = std::abs(analytical[row][parameter]
                                                 - finiteDifference[row][parameter]);
                const double relative = difference / scale;
                require(difference < 1.0e-4 || relative < 5.0e-3,
                        "Analytical Jacobian differs from finite difference: analytical="
                        + std::to_string(analytical[row][parameter]) + ", finite="
                        + std::to_string(finiteDifference[row][parameter]) + ", relative="
                        + std::to_string(relative));
            }
        }
    }
}

void testJointMomentInversion()
{
    pytem::ForwardModel low;
    low.thicknesses = {15.0};
    low.resistivities = {80.0, 20.0};
    low.times = {1.0e-5, 3.0e-5, 8.0e-5};
    low.txSize = 12.5;
    pytem::ForwardModel high = low;
    high.times = {1.0e-4, 3.0e-4, 8.0e-4};
    const auto lowObserved = pytem::TemSolver::forward(low);
    const auto highObserved = pytem::TemSolver::forward(high);

    pytem::InversionOptions options;
    options.model = low;
    options.model.resistivities = {50.0, 50.0};
    options.observed = lowObserved;
    options.noiseStd.resize(lowObserved.size());
    for (std::size_t i = 0; i < lowObserved.size(); ++i)
        options.noiseStd[i] = lowObserved[i] * 0.05;
    pytem::InversionDataSet additional;
    additional.model = high;
    additional.model.resistivities = options.model.resistivities;
    additional.observed = highObserved;
    additional.noiseStd.resize(highObserved.size());
    for (std::size_t i = 0; i < highObserved.size(); ++i)
        additional.noiseStd[i] = highObserved[i] * 0.05;
    options.additionalDataSets.push_back(additional);
    options.maxIterations = 1;
    options.alphaSteps = 2;
    options.calculateSensitivity = false;
    const auto result = pytem::TemSolver::invert(options);
    require(result.predicted.size() == lowObserved.size() + highObserved.size(),
            "Joint inversion did not return both moment predictions");
}

void testFirstJacobianCacheAndL1()
{
    pytem::ForwardModel truth;
    truth.thicknesses = {15.0};
    truth.resistivities = {100.0, 15.0};
    truth.times = {1.0e-5, 3.0e-5, 1.0e-4};
    truth.txSize = 12.5;
    const auto observed = pytem::TemSolver::forward(truth);

    pytem::InversionOptions options;
    options.model = truth;
    options.model.resistivities = {50.0, 50.0};
    options.observed = observed;
    options.noiseStd.resize(observed.size());
    for (std::size_t i = 0; i < observed.size(); ++i)
        options.noiseStd[i] = observed[i] * 0.01;
    options.maxIterations = 1;
    options.alphaSteps = 2;
    options.calculateSensitivity = false;
    options.regularizationNorm = pytem::RegularizationNorm::L1Blocky;
    options.cacheFirstJacobian = true;
    const auto unique = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path()
        / ("pytem_jacobian_test_" + std::to_string(unique));
    options.jacobianCacheDirectory = directory.string();

    pytem::TemSolver::clearJacobianMemoryCache();
    std::vector<std::future<pytem::InversionResult>> futures;
    for (int i = 0; i < 4; ++i)
        futures.push_back(std::async(std::launch::async, [options]() {
            return pytem::TemSolver::invert(options);
        }));
    int computedCount = 0;
    int memoryCount = 0;
    for (auto &future : futures) {
        const auto result = future.get();
        computedCount += result.firstJacobianSource == "computed and cached" ? 1 : 0;
        memoryCount += result.firstJacobianSource == "memory cache" ? 1 : 0;
    }
    require(computedCount == 1 && memoryCount == 3,
            "Parallel inversions did not share one first Jacobian");
    pytem::TemSolver::clearJacobianMemoryCache();
    const auto disk = pytem::TemSolver::invert(options);
    require(disk.firstJacobianSource == "disk cache",
            "First Jacobian was not reused from disk");
    pytem::TemSolver::clearJacobianMemoryCache();
    options.model.txSize += 0.1;
    const auto invalidated = pytem::TemSolver::invert(options);
    require(invalidated.firstJacobianSource == "computed and cached",
            "An incompatible first-Jacobian cache entry was reused");
    std::error_code cleanupError;
    std::filesystem::remove_all(directory, cleanupError);
}

void testMatrixIdentityShape()
{
    const std::vector<double> gates{1.2e-5, 1.8e-5, 2.8e-5};
    const std::vector<double> opens{1.0e-5, 1.5e-5, 2.4e-5};
    const std::vector<double> closes{1.4e-5, 2.1e-5, 3.2e-5};
    const std::vector<double> waveformTimes{-5.0e-6, 0.0, 6.0e-6};
    const std::vector<double> waveformAmplitudes{-1.0, 1.0, 0.0};
    const auto convolution = pytem::MatrixConvolution::build(
        gates, opens, closes, waveformTimes, waveformAmplitudes, 80, 21, 1.0);
    require(convolution.matrix.size() == gates.size(), "Matrix row count is wrong");
    require(convolution.stepTimes.size() == 80, "Step grid size is wrong");
    for (const auto &row : convolution.matrix)
        require(row.size() == 80, "Matrix column count is wrong");
}

void testUsfReader(const std::string &path)
{
    if (path.empty()) return;
    const auto sounding = pytem::UsfReader::read(path);
    require(sounding.moments.size() == 2, "Expected LM and HM moments");
    require(!sounding.moments[0].times.empty() && !sounding.moments[1].times.empty(),
            "Parsed moments must contain gates");
    require(sounding.moments[0].lowPassFrequencies.size() == 2,
            "Expected two USF low-pass stages");
    require(std::abs(sounding.equivalentCircularRadius() - std::sqrt(9.0 / 3.141592653589793)) < 1e-12,
            "Equivalent circular radius is wrong");
    if (path.find("L008_S001") != std::string::npos) {
        require(sounding.moments[0].times.size() == 8, "Expected eight LM gates");
        require(sounding.moments[1].times.size() == 21, "Expected 21 HM gates");
        require(sounding.moments[0].stackCount + sounding.moments[1].stackCount == 344,
                "Unexpected USF stack count");
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        testForwardReference();
        testEulerReference();
        testAnalyticalJacobian();
        testJointMomentInversion();
        testFirstJacobianCacheAndL1();
        testMatrixIdentityShape();
        testUsfReader(argc > 1 ? argv[1] : "");
        std::cout << "All native C++ tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
