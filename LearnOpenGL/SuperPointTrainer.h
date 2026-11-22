#pragma once

#include <glad.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ComputeShader.h"
#include "SuperPoint.h"


//-----------Configuration---------//

struct SuperPointTrainerConfig
{
    float learningRate = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    std::string shaderDirectory = "Resources/Shaders/";

    bool verbose = true;
};


//-----------Data---------//

// One training crop. The pointers are NOT owned; they must stay valid during the call.
struct TrainingSample
{
    const unsigned char* rgba = nullptr;   // width * height * 4 bytes, same size as SuperPoint::initialize()
    const int* labels = nullptr;           // (height / 8) * (width / 8) values, 0..63 = pixel in cell, 64 = no keypoint
};

struct TrainingStats
{
    float loss = 0.0f;          // mean cross-entropy per cell
    float hit = 0.0f;           // labelled cells where the top answer is exactly the labelled pixel
    float falseAlarm = 0.0f;    // unlabelled cells where the top answer is a keypoint anyway
};


//-----------SuperPointTrainer---------//
//
// Trains ONLY the detector head (convPa + convPb) of an initialized SuperPoint.
// The encoder and the descriptor head are never touched, so the descriptors (and
// therefore the pretrained SuperGlue) stay exactly the same.
//
// The trainer updates the weight SSBOs owned by the SuperPoint object IN PLACE:
// after a step, SuperPoint::extract() already uses the new detector weights.
//
// Same rules as SuperPoint: every method needs the OpenGL context to be current,
// and the SuperPoint must be initialized (initialize(width, height)) BEFORE
// SuperPointTrainer::initialize().

class SuperPointTrainer
{
public:

    SuperPointTrainer(SuperPoint& network, const SuperPointTrainerConfig& config = SuperPointTrainerConfig());

    SuperPointTrainer(const SuperPointTrainer&) = delete;
    SuperPointTrainer& operator=(const SuperPointTrainer&) = delete;

    ~SuperPointTrainer();


    bool initialize();

    bool isReady() const { return ready; }

    const SuperPointTrainerConfig& getConfig() const { return config; }

    int getStepCount() const { return step; }


    // One optimizer step on a batch (gradients are averaged over every cell of the batch,
    // exactly like torch's cross_entropy(reduction='mean') on a (B, 65, h, w) tensor).
    bool trainStep(const std::vector<TrainingSample>& batch, TrainingStats& stats);

    // Forward pass only, statistics over all the samples.
    bool evaluate(const std::vector<TrainingSample>& samples, TrainingStats& stats);

    // Writes ALL 12 layers in the weights.bin layout, so SuperPointConfig::weightsPath can load it.
    bool saveWeights(const std::string& path) const;

    // Debug: loss + gradients of one sample (no update) printed as norm / first values.
    // Compare with tools/GradCheck.py to validate the backward shaders.
    bool printGradientSummary(const TrainingSample& sample);


private:

    struct Parameter
    {
        GLuint value = 0;     // owned by SuperPoint
        GLuint grad = 0;
        GLuint m = 0;
        GLuint v = 0;
        size_t count = 0;
        const char* name = "";
    };

    bool createShaders();
    bool createBuffers();
    void releaseResources();

    void zeroGradients();
    void zeroStats();

    bool processSample(const TrainingSample& sample, float invScale, bool backward, double& lossSum);

    void runForward(GLuint features);
    void runLoss(float invScale);
    void runBackward(GLuint features);
    void runAdam();

    TrainingStats readStats(double lossSum, size_t cellCount) const;

    static std::vector<float> download(GLuint buffer, size_t count);


    SuperPoint& net;
    SuperPointTrainerConfig config;

    bool ready = false;

    int cells = 0;          // (width / 8) * (height / 8)
    int width3 = 0;
    int height3 = 0;
    int step = 0;

    std::unique_ptr<ComputeShader> forwardAShader;
    std::unique_ptr<ComputeShader> forwardBShader;
    std::unique_ptr<ComputeShader> lossShader;
    std::unique_ptr<ComputeShader> backwardBShader;
    std::unique_ptr<ComputeShader> backwardHiddenShader;
    std::unique_ptr<ComputeShader> backwardAShader;
    std::unique_ptr<ComputeShader> adamShader;

    // Activations / gradients of one image (reused for every sample)
    GLuint hiddenBuffer = 0;     // 256 x cells
    GLuint logitsBuffer = 0;     // 65 x cells
    GLuint dLogitsBuffer = 0;    // 65 x cells
    GLuint dHiddenBuffer = 0;    // 256 x cells
    GLuint labelBuffer = 0;      // cells ints
    GLuint lossBuffer = 0;       // cells floats
    GLuint statsBuffer = 0;      // 4 uints

    // 0 = convPa weights, 1 = convPa bias, 2 = convPb weights, 3 = convPb bias
    Parameter parameters[4];

};
