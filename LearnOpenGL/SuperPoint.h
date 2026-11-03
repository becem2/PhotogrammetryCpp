#pragma once

#include <glad.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ComputeShader.h"
#include "Texture.h"


//-----------Configuration---------//

struct SuperPointConfig
{
    int nmsRadius = 4;
    int borderSize = 4;

    float detectionThreshold = 0.005f;

    GLuint maxKeypoints = 4096;

    std::string shaderDirectory = "Resources/Shaders/";
    std::string weightsPath = "Resources/Weights/weights.bin";

    bool verbose = true;
};


//-----------Output---------//

struct Keypoint
{
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
};

struct SuperPointFeatures
{
    static constexpr unsigned int DescriptorDimension = 256;

    int imageWidth = 0;
    int imageHeight = 0;

    // Sorted by descending score
    std::vector<Keypoint> keypoints;

    // keypoints.size() * 256 floats, keypoint-major, each descriptor L2-normalized
    std::vector<float> descriptors;

    size_t count() const { return keypoints.size(); }
};


//-----------SuperPoint (OpenGL compute pipeline)---------//
//
// Every method that touches OpenGL (constructor excluded, initialize, extract and
// the destructor) must be called while the OpenGL context is current.
// Destroy the object BEFORE destroying the window / context.

class SuperPoint
{
public:

    explicit SuperPoint(const SuperPointConfig& config = SuperPointConfig());

    SuperPoint(const SuperPoint&) = delete;
    SuperPoint& operator=(const SuperPoint&) = delete;

    ~SuperPoint();


    // Allocates textures, compiles shaders, loads weights, creates buffers.
    // width and height must be positive multiples of 8.
    bool initialize(int width, int height);

    bool isReady() const { return ready; }

    int getWidth() const { return imageWidth; }
    int getHeight() const { return imageHeight; }

    const SuperPointConfig& getConfig() const { return config; }


    // rgbaPixels: width * height * 4 bytes (what stbi_load(..., 4) returns),
    // exactly the size passed to initialize().
    bool extract(const unsigned char* rgbaPixels, SuperPointFeatures& features);


    // Binary format: [magic, version, N, 256] (uint32), N x (x, y, score) floats,
    // then N x 256 descriptor floats.
    static bool save(const std::string& path, const SuperPointFeatures& features);


private:

    struct ConvUniforms
    {
        GLint inputChannels = -1;
        GLint outputChannels = -1;
        GLint kernelSize = -1;
    };

    bool createTextures();
    bool createShaders();
    bool queryUniforms();
    bool loadWeights();
    bool createKeypointBuffers();
    void releaseResources();

    void runConv(int layer, GLuint inputTexture, GLuint outputTexture,
        bool relu, int width, int height);

    void runMaxPool(GLuint inputTexture, GLuint outputTexture,
        int outputWidth, int outputHeight, int channels);

    void runNmsDilate(GLuint maskTexture, GLuint suppressionTexture);

    void runNmsUpdate(GLuint currentMask, GLuint suppressionTexture, GLuint nextMask);


    SuperPointConfig config;

    bool ready = false;

    int imageWidth = 0;
    int imageHeight = 0;

    int width1 = 0, height1 = 0;
    int width2 = 0, height2 = 0;
    int width3 = 0, height3 = 0;


    // Textures
    std::unique_ptr<Texture> inputTexture;
    std::unique_ptr<Texture> grayTexture;
    std::unique_ptr<Texture> tensorTexture;
    std::unique_ptr<Texture> pingA;
    std::unique_ptr<Texture> pingB;
    std::unique_ptr<Texture> descriptorTexture;
    std::unique_ptr<Texture> detectorTexture;
    std::unique_ptr<Texture> heatmapTexture;
    std::unique_ptr<Texture> nmsMaskA;
    std::unique_ptr<Texture> nmsMaskB;
    std::unique_ptr<Texture> nmsSuppression;
    std::unique_ptr<Texture> filteredScoreTexture;

    // Shaders
    std::unique_ptr<ComputeShader> grayShader;
    std::unique_ptr<ComputeShader> toTensorShader;
    std::unique_ptr<ComputeShader> conv1aShader;
    std::unique_ptr<ComputeShader> convReLUShader;
    std::unique_ptr<ComputeShader> convShader;
    std::unique_ptr<ComputeShader> maxPoolShader;
    std::unique_ptr<ComputeShader> descriptorL2Shader;
    std::unique_ptr<ComputeShader> softmaxShader;
    std::unique_ptr<ComputeShader> nmsInitShader;
    std::unique_ptr<ComputeShader> nmsDilateShader;
    std::unique_ptr<ComputeShader> nmsUpdateShader;
    std::unique_ptr<ComputeShader> thresholdShader;
    std::unique_ptr<ComputeShader> compactShader;
    std::unique_ptr<ComputeShader> sampleShader;

    // Uniform locations
    ConvUniforms convReLUUniforms;
    ConvUniforms convUniforms;

    GLint epsilonLocation = -1;
    GLint nmsInitRadiusLocation = -1;
    GLint nmsDilateRadiusLocation = -1;
    GLint nmsUpdateRadiusLocation = -1;
    GLint thresholdLocation = -1;
    GLint borderLocation = -1;
    GLint keypointCapacityLocation = -1;
    GLint sampleKeypointCountLocation = -1;

    // Buffers
    static constexpr int LayerCount = 12;

    GLuint weightsBuffer[LayerCount] = {};
    GLuint biasesBuffer[LayerCount] = {};
    bool weightBuffersCreated = false;

    GLuint keypointCounterBuffer = 0;
    GLuint keypointCandidateBuffer = 0;
    GLuint sampledDescriptorBuffer = 0;
};