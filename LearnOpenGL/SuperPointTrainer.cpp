#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "SuperPointTrainer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>


namespace
{
    constexpr int kEncoderChannels = 128;
    constexpr int kHiddenChannels = 256;
    constexpr int kClasses = 65;

    constexpr int kLocal1D = 64;
    constexpr int kLocal2D = 16;
    constexpr int kAdamLocal = 256;
    constexpr int kCellsPerChunk = 4800;      // 80 x 60 cells = a 640 x 480 crop

    inline GLuint groups(int n, int localSize)
    {
        return static_cast<GLuint>((n + localSize - 1) / localSize);
    }

    inline void storageBarrier()
    {
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    }

    inline void bindStorage(GLuint index, GLuint buffer)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer);
    }

    inline void bindFeatures(GLuint texture)
    {
        glBindImageTexture(0, texture, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    }

    GLuint createBuffer(size_t bytes)
    {
        GLuint buffer = 0;
        glGenBuffers(1, &buffer);

        if (buffer == 0) return 0;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_DYNAMIC_COPY);

        return buffer;
    }

    // All-zero bits are 0.0f and 0u, so this clears float and uint buffers alike.
    void clearBuffer(GLuint buffer)
    {
        const float zero = 0.0f;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &zero);
    }
}


//-----------Construction / Destruction---------//

SuperPointTrainer::SuperPointTrainer(SuperPoint& network, const SuperPointTrainerConfig& cfg)
    : net(network), config(cfg)
{
}


SuperPointTrainer::~SuperPointTrainer()
{
    releaseResources();
}


void SuperPointTrainer::releaseResources()
{
    ready = false;

    forwardAShader.reset();
    forwardBShader.reset();
    lossShader.reset();
    backwardBShader.reset();
    backwardHiddenShader.reset();
    backwardAShader.reset();
    adamShader.reset();

    GLuint* singles[] = {
        &hiddenBuffer, &logitsBuffer, &dLogitsBuffer, &dHiddenBuffer,
        &labelBuffer, &lossBuffer, &statsBuffer
    };

    for (GLuint* b : singles) {
        if (*b != 0) {
            glDeleteBuffers(1, b);
            *b = 0;
        }
    }

    for (Parameter& p : parameters) {
        // p.value belongs to SuperPoint, never delete it here
        GLuint* owned[] = { &p.grad, &p.m, &p.v };

        for (GLuint* b : owned) {
            if (*b != 0) {
                glDeleteBuffers(1, b);
                *b = 0;
            }
        }

        p.value = 0;
        p.count = 0;
    }
}


//-----------Initialization---------//

bool SuperPointTrainer::initialize()
{
    releaseResources();

    if (!net.isReady()) {
        std::cerr << "SuperPointTrainer: the SuperPoint must be initialized first." << std::endl;
        return false;
    }

    width3 = net.getWidth() / 8;
    height3 = net.getHeight() / 8;
    cells = width3 * height3;
    step = 0;

    if (!createShaders() || !createBuffers()) {
        releaseResources();
        return false;
    }

    ready = true;

    return true;
}


bool SuperPointTrainer::createShaders()
{
    const std::filesystem::path dir(config.shaderDirectory);

    auto load = [&](const char* fileName) -> std::unique_ptr<ComputeShader>
        {
            auto shader = std::make_unique<ComputeShader>((dir / fileName).string());

            if (shader->getProgram() == 0) {
                std::cerr << "SuperPointTrainer: failed to build shader '" << fileName << "'" << std::endl;
                return nullptr;
            }

            return shader;
        };

    forwardAShader = load("SPT_HeadForwardA.comp.txt");
    forwardBShader = load("SPT_HeadForwardB.comp.txt");
    lossShader = load("SPT_LossGrad.comp.txt");
    backwardBShader = load("SPT_BackwardB.comp.txt");
    backwardHiddenShader = load("SPT_BackwardHidden.comp.txt");
    backwardAShader = load("SPT_BackwardA.comp.txt");
    adamShader = load("SPT_AdamUpdate.comp.txt");

    return forwardAShader && forwardBShader && lossShader && backwardBShader &&
        backwardHiddenShader && backwardAShader && adamShader;
}


bool SuperPointTrainer::createBuffers()
{
    const size_t n = static_cast<size_t>(cells);

    GLint64 maxSSBOSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxSSBOSize);

    const size_t largest = n * kHiddenChannels * sizeof(float);

    if (largest > static_cast<size_t>(maxSSBOSize)) {
        std::cerr << "SuperPointTrainer: the hidden activation buffer (" << largest / (1024.0 * 1024.0)
            << " MB) exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE." << std::endl;
        return false;
    }

    hiddenBuffer = createBuffer(n * kHiddenChannels * sizeof(float));
    dHiddenBuffer = createBuffer(n * kHiddenChannels * sizeof(float));
    logitsBuffer = createBuffer(n * kClasses * sizeof(float));
    dLogitsBuffer = createBuffer(n * kClasses * sizeof(float));
    labelBuffer = createBuffer(n * sizeof(int));
    lossBuffer = createBuffer(n * sizeof(float));
    statsBuffer = createBuffer(4 * sizeof(unsigned int));

    if (!hiddenBuffer || !dHiddenBuffer || !logitsBuffer || !dLogitsBuffer ||
        !labelBuffer || !lossBuffer || !statsBuffer) {
        std::cerr << "SuperPointTrainer: failed to allocate activation buffers." << std::endl;
        return false;
    }

    // Parameters: the value buffers are the ones SuperPoint uses for inference
    const int pa = SuperPoint::ConvPaLayer;
    const int pb = SuperPoint::ConvPbLayer;

    parameters[0] = { net.weightBuffer(pa), 0, 0, 0, SuperPoint::layerWeightCount(pa), "convPa.weight" };
    parameters[1] = { net.biasBuffer(pa),   0, 0, 0, SuperPoint::layerBiasCount(pa),   "convPa.bias" };
    parameters[2] = { net.weightBuffer(pb), 0, 0, 0, SuperPoint::layerWeightCount(pb), "convPb.weight" };
    parameters[3] = { net.biasBuffer(pb),   0, 0, 0, SuperPoint::layerBiasCount(pb),   "convPb.bias" };

    size_t total = 0;

    for (Parameter& p : parameters) {
        if (p.value == 0) {
            std::cerr << "SuperPointTrainer: SuperPoint weight buffers are missing." << std::endl;
            return false;
        }

        p.grad = createBuffer(p.count * sizeof(float));
        p.m = createBuffer(p.count * sizeof(float));
        p.v = createBuffer(p.count * sizeof(float));

        if (!p.grad || !p.m || !p.v) {
            std::cerr << "SuperPointTrainer: failed to allocate optimizer buffers." << std::endl;
            return false;
        }

        clearBuffer(p.grad);
        clearBuffer(p.m);
        clearBuffer(p.v);

        total += p.count;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    storageBarrier();

    if (config.verbose) {
        const double activationMB = (2.0 * n * kHiddenChannels + 2.0 * n * kClasses) * sizeof(float) / (1024.0 * 1024.0);

        std::cout << "SuperPointTrainer: training " << total << " weights (convPa + convPb), "
            << cells << " cells per image, " << activationMB << " MB of activations." << std::endl;
    }

    return true;
}


//-----------Small Helpers---------//

void SuperPointTrainer::zeroGradients()
{
    for (Parameter& p : parameters) {
        clearBuffer(p.grad);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    storageBarrier();
}


void SuperPointTrainer::zeroStats()
{
    clearBuffer(statsBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    storageBarrier();
}


std::vector<float> SuperPointTrainer::download(GLuint buffer, size_t count)
{
    std::vector<float> data(count);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(count * sizeof(float)), data.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return data;
}


TrainingStats SuperPointTrainer::readStats(double lossSum, size_t cellCount) const
{
    unsigned int s[4] = {};

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(s), s);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    TrainingStats stats;
    stats.loss = static_cast<float>(lossSum / static_cast<double>(std::max<size_t>(cellCount, 1)));
    stats.hit = static_cast<float>(s[0]) / static_cast<float>(std::max(s[1], 1u));
    stats.falseAlarm = static_cast<float>(s[2]) / static_cast<float>(std::max(s[3], 1u));

    return stats;
}


//-----------Network Passes---------//

void SuperPointTrainer::runForward(GLuint features)
{
    // convPa : 3x3, 128 -> 256, ReLU
    forwardAShader->use();

    bindFeatures(features);
    bindStorage(0, net.weightBuffer(SuperPoint::ConvPaLayer));
    bindStorage(1, net.biasBuffer(SuperPoint::ConvPaLayer));
    bindStorage(2, hiddenBuffer);

    glDispatchCompute(groups(width3, kLocal2D), groups(height3, kLocal2D), kHiddenChannels);
    storageBarrier();

    // convPb : 1x1, 256 -> 65
    forwardBShader->use();

    glUniform1i(0, cells);

    bindStorage(0, net.weightBuffer(SuperPoint::ConvPbLayer));
    bindStorage(1, net.biasBuffer(SuperPoint::ConvPbLayer));
    bindStorage(2, hiddenBuffer);
    bindStorage(3, logitsBuffer);

    glDispatchCompute(groups(cells, kLocal1D), kClasses, 1);
    storageBarrier();
}


void SuperPointTrainer::runLoss(float invScale)
{
    lossShader->use();

    glUniform1i(0, cells);
    glUniform1f(1, invScale);

    bindStorage(0, logitsBuffer);
    bindStorage(1, labelBuffer);
    bindStorage(2, dLogitsBuffer);
    bindStorage(3, lossBuffer);
    bindStorage(4, statsBuffer);

    glDispatchCompute(groups(cells, kLocal1D), 1, 1);
    storageBarrier();
}


void SuperPointTrainer::runBackward(GLuint features)
{
    // convPb weights + bias
    backwardBShader->use();

    glUniform1i(0, cells);

    bindStorage(0, dLogitsBuffer);
    bindStorage(1, hiddenBuffer);
    bindStorage(2, parameters[2].grad);
    bindStorage(3, parameters[3].grad);

    glDispatchCompute(groups(kClasses * kHiddenChannels, kLocal1D), 1, 1);
    storageBarrier();

    // Back through convPb and the ReLU
    backwardHiddenShader->use();

    glUniform1i(0, cells);

    bindStorage(0, net.weightBuffer(SuperPoint::ConvPbLayer));
    bindStorage(1, dLogitsBuffer);
    bindStorage(2, hiddenBuffer);
    bindStorage(3, dHiddenBuffer);

    glDispatchCompute(groups(cells, kLocal1D), kHiddenChannels, 1);
    storageBarrier();

    // convPa weights + bias
    backwardAShader->use();

    bindFeatures(features);
    bindStorage(0, dHiddenBuffer);
    bindStorage(1, parameters[0].grad);
    bindStorage(2, parameters[1].grad);

    // The cell rows are split in chunks of about kCellsPerChunk cells: every dispatch then costs
    // the same as a whole 640 x 480 crop, however large the crop is.
    const int rowsPerChunk = std::max(1, kCellsPerChunk / std::max(width3, 1));

    for (int row = 0; row < height3; row += rowsPerChunk) {

        glUniform1i(0, row);
        glUniform1i(1, std::min(row + rowsPerChunk, height3));

        glDispatchCompute(groups(kHiddenChannels * kEncoderChannels * 9, kLocal1D), 1, 1);
        storageBarrier();

        if (rowsPerChunk < height3) {
            glFinish();      // end the GPU command here: keeps each one short for the Windows watchdog
        }
    }
}


void SuperPointTrainer::runAdam()
{
    step++;

    const float bias1 = 1.0f - std::pow(config.beta1, static_cast<float>(step));
    const float bias2 = 1.0f - std::pow(config.beta2, static_cast<float>(step));

    adamShader->use();

    glUniform1f(1, config.learningRate);
    glUniform1f(2, config.beta1);
    glUniform1f(3, config.beta2);
    glUniform1f(4, config.epsilon);
    glUniform1f(5, 1.0f / bias1);
    glUniform1f(6, 1.0f / bias2);

    for (const Parameter& p : parameters) {

        glUniform1i(0, static_cast<GLint>(p.count));

        bindStorage(0, p.value);
        bindStorage(1, p.grad);
        bindStorage(2, p.m);
        bindStorage(3, p.v);

        glDispatchCompute(groups(static_cast<int>(p.count), kAdamLocal), 1, 1);
        storageBarrier();
    }
}


//-----------One Sample---------//

bool SuperPointTrainer::processSample(const TrainingSample& sample, float invScale,
    bool backward, double& lossSum)
{
    if (sample.rgba == nullptr || sample.labels == nullptr) {
        std::cerr << "SuperPointTrainer: sample without image or labels." << std::endl;
        return false;
    }

    // Shared encoder (frozen): leaves the 128-channel 1/8 feature map in SuperPoint
    if (!net.encode(sample.rgba)) {
        return false;
    }

    const GLuint features = net.encoderFeatureTexture();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, labelBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(cells * sizeof(int)), sample.labels);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    runForward(features);
    runLoss(invScale);

    const std::vector<float> perCell = download(lossBuffer, static_cast<size_t>(cells));

    for (float l : perCell) {
        lossSum += l;
    }

    if (backward) {
        runBackward(features);
    }

    return true;
}


//-----------Public API---------//

bool SuperPointTrainer::trainStep(const std::vector<TrainingSample>& batch, TrainingStats& stats)
{
    stats = TrainingStats();

    if (!ready || batch.empty()) {
        std::cerr << "SuperPointTrainer: trainStep() needs initialize() and a non-empty batch." << std::endl;
        return false;
    }

    zeroGradients();
    zeroStats();

    const size_t totalCells = batch.size() * static_cast<size_t>(cells);
    const float invScale = 1.0f / static_cast<float>(totalCells);

    double lossSum = 0.0;

    for (const TrainingSample& sample : batch) {
        if (!processSample(sample, invScale, true, lossSum)) {
            return false;
        }
    }

    runAdam();

    stats = readStats(lossSum, totalCells);

    return true;
}


bool SuperPointTrainer::evaluate(const std::vector<TrainingSample>& samples, TrainingStats& stats)
{
    stats = TrainingStats();

    if (!ready || samples.empty()) {
        std::cerr << "SuperPointTrainer: evaluate() needs initialize() and at least one sample." << std::endl;
        return false;
    }

    zeroStats();

    double lossSum = 0.0;

    for (const TrainingSample& sample : samples) {
        if (!processSample(sample, 1.0f, false, lossSum)) {
            return false;
        }
    }

    stats = readStats(lossSum, samples.size() * static_cast<size_t>(cells));

    return true;
}


bool SuperPointTrainer::saveWeights(const std::string& path) const
{
    std::ofstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "SuperPointTrainer: could not create '" << path << "'" << std::endl;
        return false;
    }

    for (int l = 0; l < SuperPoint::LayerCount; l++) {

        const std::vector<float> w = download(net.weightBuffer(l), SuperPoint::layerWeightCount(l));
        const std::vector<float> b = download(net.biasBuffer(l), SuperPoint::layerBiasCount(l));

        file.write(reinterpret_cast<const char*>(w.data()), static_cast<std::streamsize>(w.size() * sizeof(float)));
        file.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size() * sizeof(float)));
    }

    file.close();

    return static_cast<bool>(file);
}


bool SuperPointTrainer::printGradientSummary(const TrainingSample& sample)
{
    if (!ready) return false;

    zeroGradients();
    zeroStats();

    double lossSum = 0.0;

    if (!processSample(sample, 1.0f / static_cast<float>(cells), true, lossSum)) {
        return false;
    }

    const TrainingStats stats = readStats(lossSum, static_cast<size_t>(cells));

    std::cout << "loss " << stats.loss << " | hit " << stats.hit * 100.0f
        << "% | false alarm " << stats.falseAlarm * 100.0f << "%" << std::endl;

    for (const Parameter& p : parameters) {

        const std::vector<float> g = download(p.grad, p.count);

        double norm = 0.0;
        for (float v : g) norm += static_cast<double>(v) * v;

        std::cout << "grad " << p.name << " : L2 norm " << std::sqrt(norm) << " | first values";

        for (size_t i = 0; i < std::min<size_t>(4, g.size()); i++) {
            std::cout << " " << g[i];
        }

        std::cout << std::endl;
    }

    return true;
}
