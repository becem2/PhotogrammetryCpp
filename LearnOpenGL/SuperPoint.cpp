#include "SuperPoint.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <tuple>


namespace
{
    //-----------Network Description (must match weights.bin order)---------//

    enum LayerIndex
    {
        CONV1A = 0,
        CONV1B,
        CONV2A,
        CONV2B,
        CONV3A,
        CONV3B,
        CONV4A,
        CONV4B,
        CONVDA,
        CONVDB,
        CONVPA,
        CONVPB
    };

    const char* const kLayerNames[12] = {
        "conv1a", "conv1b", "conv2a", "conv2b",
        "conv3a", "conv3b", "conv4a", "conv4b",
        "convDa", "convDb", "convPa", "convPb"
    };

    constexpr int kLayerCin[12] = { 1, 64, 64, 64, 64, 128, 128, 128, 128, 256, 128, 256 };
    constexpr int kLayerCout[12] = { 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 65 };
    constexpr int kLayerK[12] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1 };

    constexpr int kLocalSize = 16;
    constexpr int kSampleLocalSize = 64;

    constexpr int kDescriptorDimension = 256;
    constexpr int kDetectorChannels = 65;


    inline GLuint groups(int n, int localSize = kLocalSize)
    {
        return static_cast<GLuint>((n + localSize - 1) / localSize);
    }

    inline void bindImage(GLuint unit, GLuint texture, GLenum access, GLenum format, bool layered)
    {
        glBindImageTexture(unit, texture, 0, layered ? GL_TRUE : GL_FALSE, 0, access, format);
    }

    inline void imageBarrier()
    {
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
}


//-----------Construction / Destruction---------//

SuperPoint::SuperPoint(const SuperPointConfig& cfg)
    : config(cfg)
{
}


SuperPoint::~SuperPoint()
{
    releaseResources();
}


void SuperPoint::releaseResources()
{
    ready = false;

    // Textures and shaders clean themselves up through their destructors
    inputTexture.reset();
    grayTexture.reset();
    tensorTexture.reset();
    pingA.reset();
    pingB.reset();
    descriptorTexture.reset();
    detectorTexture.reset();
    heatmapTexture.reset();
    nmsMaskA.reset();
    nmsMaskB.reset();
    nmsSuppression.reset();
    filteredScoreTexture.reset();

    grayShader.reset();
    toTensorShader.reset();
    conv1aShader.reset();
    convReLUShader.reset();
    convShader.reset();
    maxPoolShader.reset();
    descriptorL2Shader.reset();
    softmaxShader.reset();
    nmsInitShader.reset();
    nmsDilateShader.reset();
    nmsUpdateShader.reset();
    thresholdShader.reset();
    compactShader.reset();
    sampleShader.reset();

    if (weightBuffersCreated) {
        glDeleteBuffers(LayerCount, weightsBuffer);
        glDeleteBuffers(LayerCount, biasesBuffer);

        std::fill(std::begin(weightsBuffer), std::end(weightsBuffer), 0u);
        std::fill(std::begin(biasesBuffer), std::end(biasesBuffer), 0u);

        weightBuffersCreated = false;
    }

    if (keypointCounterBuffer != 0) {
        glDeleteBuffers(1, &keypointCounterBuffer);
        keypointCounterBuffer = 0;
    }

    if (keypointCandidateBuffer != 0) {
        glDeleteBuffers(1, &keypointCandidateBuffer);
        keypointCandidateBuffer = 0;
    }

    if (sampledDescriptorBuffer != 0) {
        glDeleteBuffers(1, &sampledDescriptorBuffer);
        sampledDescriptorBuffer = 0;
    }
}


//-----------Initialization---------//

bool SuperPoint::initialize(int width, int height)
{
    releaseResources();

    //-----------Validate Dimensions---------//

    if (width <= 0 || height <= 0 || width % 8 != 0 || height % 8 != 0) {
        std::cerr << "SuperPoint: image dimensions must be positive and divisible by 8." << std::endl;
        return false;
    }

    GLint maxTextureSize = 0;
    GLint maxArrayLayers = 0;

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxArrayLayers);

    if (width > maxTextureSize || height > maxTextureSize || maxArrayLayers < kDescriptorDimension) {
        std::cerr << "SuperPoint: image dimensions or array layer count exceed OpenGL limits."
            << " (max texture size " << maxTextureSize
            << ", max array layers " << maxArrayLayers << ")" << std::endl;
        return false;
    }

    imageWidth = width;
    imageHeight = height;

    width1 = width / 2;   height1 = height / 2;
    width2 = width / 4;   height2 = height / 4;
    width3 = width / 8;   height3 = height / 8;


    //-----------Check SSBO Limits---------//

    const size_t pixelCount = static_cast<size_t>(width) * height;

    const size_t candidateBufferSize = pixelCount * sizeof(GLuint) * 2;

    const size_t descriptorBufferSize =
        static_cast<size_t>(config.maxKeypoints) * kDescriptorDimension * sizeof(float);

    GLint64 maxSSBOSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxSSBOSize);

    if (candidateBufferSize > static_cast<size_t>(maxSSBOSize) ||
        descriptorBufferSize > static_cast<size_t>(maxSSBOSize)) {

        std::cerr << "SuperPoint: required SSBO size exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE ("
            << maxSSBOSize / (1024.0 * 1024.0) << " MB)." << std::endl;
        return false;
    }


    //-----------Create Resources---------//

    if (!createTextures() ||
        !createShaders() ||
        !queryUniforms() ||
        !loadWeights() ||
        !createKeypointBuffers()) {

        releaseResources();
        return false;
    }

    ready = true;

    return true;
}


bool SuperPoint::createTextures()
{
    const unsigned int W = static_cast<unsigned int>(imageWidth);
    const unsigned int H = static_cast<unsigned int>(imageHeight);
    const unsigned int W3 = static_cast<unsigned int>(width3);
    const unsigned int H3 = static_cast<unsigned int>(height3);

    inputTexture = std::make_unique<Texture>(LayerType::NORMAL, false, GL_RGBA8, H, W, 0);
    grayTexture = std::make_unique<Texture>(LayerType::NORMAL, false, GL_RGBA8, H, W, 0);
    tensorTexture = std::make_unique<Texture>(LayerType::NORMAL, false, GL_R32F, H, W, 0);

    pingA = std::make_unique<Texture>(LayerType::NORMAL, true, GL_R32F, H, W, 64);
    pingB = std::make_unique<Texture>(LayerType::NORMAL, true, GL_R32F, H, W, 64);

    descriptorTexture = std::make_unique<Texture>(LayerType::NORMAL, true, GL_R32F, H3, W3, kDescriptorDimension);
    detectorTexture = std::make_unique<Texture>(LayerType::NORMAL, true, GL_R32F, H3, W3, kDetectorChannels);

    heatmapTexture = std::make_unique<Texture>(LayerType::NORMAL, false, GL_R32F, H, W, 0);

    nmsMaskA = std::make_unique<Texture>(LayerType::NORMAL, false, GL_R32F, H, W, 0);
    nmsMaskB = std::make_unique<Texture>(LayerType::NORMAL, false, GL_R32F, H, W, 0);
    nmsSuppression = std::make_unique<Texture>(LayerType::NORMAL, false, GL_R32F, H, W, 0);
    filteredScoreTexture = std::make_unique<Texture>(LayerType::NORMAL, false, GL_R32F, H, W, 0);

    const Texture* all[] = {
        inputTexture.get(), grayTexture.get(), tensorTexture.get(),
        pingA.get(), pingB.get(), descriptorTexture.get(), detectorTexture.get(),
        heatmapTexture.get(), nmsMaskA.get(), nmsMaskB.get(),
        nmsSuppression.get(), filteredScoreTexture.get()
    };

    for (const Texture* t : all) {
        if (t->getTextureID() == 0) {
            std::cerr << "SuperPoint: failed to create one or more textures." << std::endl;
            return false;
        }
    }

    return true;
}


bool SuperPoint::createShaders()
{
    const std::filesystem::path dir(config.shaderDirectory);

    auto load = [&](const char* fileName) -> std::unique_ptr<ComputeShader>
        {
            auto shader = std::make_unique<ComputeShader>((dir / fileName).string());

            if (shader->getProgram() == 0) {
                std::cerr << "SuperPoint: failed to build shader '" << fileName << "'" << std::endl;
                return nullptr;
            }

            return shader;
        };

    grayShader = load("RGB2Gray.comp.txt");
    toTensorShader = load("ToTensor.comp.txt");
    conv1aShader = load("Conv1aReLU.comp.txt");
    convReLUShader = load("ConvArrayReLU.comp.txt");
    convShader = load("ConvArray.comp.txt");
    maxPoolShader = load("MaxPool.comp.txt");
    descriptorL2Shader = load("DescriptorL2.comp.txt");
    softmaxShader = load("DetectorSoftmaxHeatmap.comp.txt");
    nmsInitShader = load("NMSInit.comp.txt");
    nmsDilateShader = load("NMSDilate.comp.txt");
    nmsUpdateShader = load("NMSUpdate.comp.txt");
    thresholdShader = load("ThresholdBorder.comp.txt");
    compactShader = load("KeypointCompact.comp.txt");
    sampleShader = load("SampleDescriptors.comp.txt");

    return grayShader && toTensorShader && conv1aShader && convReLUShader &&
        convShader && maxPoolShader && descriptorL2Shader && softmaxShader &&
        nmsInitShader && nmsDilateShader && nmsUpdateShader && thresholdShader &&
        compactShader && sampleShader;
}


bool SuperPoint::queryUniforms()
{
    convReLUUniforms.inputChannels = glGetUniformLocation(convReLUShader->getProgram(), "uInputChannels");
    convReLUUniforms.outputChannels = glGetUniformLocation(convReLUShader->getProgram(), "uOutputChannels");
    convReLUUniforms.kernelSize = glGetUniformLocation(convReLUShader->getProgram(), "uKernelSize");

    convUniforms.inputChannels = glGetUniformLocation(convShader->getProgram(), "uInputChannels");
    convUniforms.outputChannels = glGetUniformLocation(convShader->getProgram(), "uOutputChannels");
    convUniforms.kernelSize = glGetUniformLocation(convShader->getProgram(), "uKernelSize");

    epsilonLocation = glGetUniformLocation(descriptorL2Shader->getProgram(), "uEpsilon");

    nmsInitRadiusLocation = glGetUniformLocation(nmsInitShader->getProgram(), "uNmsRadius");
    nmsDilateRadiusLocation = glGetUniformLocation(nmsDilateShader->getProgram(), "uNmsRadius");
    nmsUpdateRadiusLocation = glGetUniformLocation(nmsUpdateShader->getProgram(), "uNmsRadius");

    thresholdLocation = glGetUniformLocation(thresholdShader->getProgram(), "uThreshold");
    borderLocation = glGetUniformLocation(thresholdShader->getProgram(), "uBorder");

    keypointCapacityLocation = glGetUniformLocation(compactShader->getProgram(), "uCapacity");
    sampleKeypointCountLocation = glGetUniformLocation(sampleShader->getProgram(), "uKeypointCount");

    const GLint locations[] = {
        convReLUUniforms.inputChannels, convReLUUniforms.outputChannels, convReLUUniforms.kernelSize,
        convUniforms.inputChannels, convUniforms.outputChannels, convUniforms.kernelSize,
        epsilonLocation,
        nmsInitRadiusLocation, nmsDilateRadiusLocation, nmsUpdateRadiusLocation,
        thresholdLocation, borderLocation,
        keypointCapacityLocation, sampleKeypointCountLocation
    };

    for (GLint location : locations) {
        if (location == -1) {
            std::cerr << "SuperPoint: one or more compute shader uniforms were not found." << std::endl;
            return false;
        }
    }

    return true;
}


bool SuperPoint::loadWeights()
{
    std::ifstream weightsFile(config.weightsPath, std::ios::binary);

    if (!weightsFile.is_open()) {
        std::cerr << "SuperPoint: failed to open weights file '" << config.weightsPath << "'" << std::endl;
        return false;
    }

    glGenBuffers(LayerCount, weightsBuffer);
    glGenBuffers(LayerCount, biasesBuffer);

    weightBuffersCreated = true;

    long long totalParameters = 0;

    for (int l = 0; l < LayerCount; l++) {

        const size_t weightCount =
            static_cast<size_t>(kLayerCout[l]) * kLayerCin[l] * kLayerK[l] * kLayerK[l];

        const size_t biasCount = static_cast<size_t>(kLayerCout[l]);

        std::vector<float> w(weightCount);
        std::vector<float> b(biasCount);

        weightsFile.read(reinterpret_cast<char*>(w.data()), weightCount * sizeof(float));
        weightsFile.read(reinterpret_cast<char*>(b.data()), biasCount * sizeof(float));

        if (!weightsFile) {
            std::cerr << "SuperPoint: weights file is too short at layer " << kLayerNames[l] << std::endl;
            return false;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightsBuffer[l]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, weightCount * sizeof(float), w.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, biasesBuffer[l]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, biasCount * sizeof(float), b.data(), GL_STATIC_DRAW);

        totalParameters += static_cast<long long>(weightCount + biasCount);

        if (config.verbose) {
            std::cout << kLayerNames[l]
                << " : weights (" << kLayerCout[l] << ", " << kLayerCin[l] << ", "
                << kLayerK[l] << ", " << kLayerK[l] << ")"
                << "  first weight : " << w[0]
                << "  first bias : " << b[0] << std::endl;
        }
    }

    if (weightsFile.peek() != EOF) {
        std::cerr << "SuperPoint WARNING: weights file has extra data after the last layer" << std::endl;
    }

    if (config.verbose) {
        std::cout << "Total parameters : " << totalParameters << std::endl;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return true;
}


bool SuperPoint::createKeypointBuffers()
{
    const size_t pixelCount = static_cast<size_t>(imageWidth) * imageHeight;

    glGenBuffers(1, &keypointCounterBuffer);
    glGenBuffers(1, &keypointCandidateBuffer);
    glGenBuffers(1, &sampledDescriptorBuffer);

    if (keypointCounterBuffer == 0 || keypointCandidateBuffer == 0 || sampledDescriptorBuffer == 0) {
        std::cerr << "SuperPoint: failed to generate keypoint SSBOs." << std::endl;
        return false;
    }

    const GLuint zero = 0;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointCounterBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointCandidateBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(pixelCount * sizeof(GLuint) * 2), NULL, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sampledDescriptorBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(static_cast<size_t>(config.maxKeypoints) * kDescriptorDimension * sizeof(float)),
        NULL, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return true;
}


//-----------Layer Helpers---------//

void SuperPoint::runConv(int layer, GLuint inputTexture, GLuint outputTexture,
    bool relu, int width, int height)
{
    const ConvUniforms& uniforms = relu ? convReLUUniforms : convUniforms;

    (relu ? convReLUShader : convShader)->use();

    glUniform1i(uniforms.inputChannels, kLayerCin[layer]);
    glUniform1i(uniforms.outputChannels, kLayerCout[layer]);
    glUniform1i(uniforms.kernelSize, kLayerK[layer]);

    bindImage(0, inputTexture, GL_READ_ONLY, GL_R32F, true);
    bindImage(1, outputTexture, GL_WRITE_ONLY, GL_R32F, true);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[layer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[layer]);

    glDispatchCompute(groups(width), groups(height), static_cast<GLuint>(kLayerCout[layer]));

    imageBarrier();
}


void SuperPoint::runMaxPool(GLuint inputTexture, GLuint outputTexture,
    int outputWidth, int outputHeight, int channels)
{
    maxPoolShader->use();

    bindImage(0, inputTexture, GL_READ_ONLY, GL_R32F, true);
    bindImage(1, outputTexture, GL_WRITE_ONLY, GL_R32F, true);

    glDispatchCompute(groups(outputWidth), groups(outputHeight), static_cast<GLuint>(channels));

    imageBarrier();
}


void SuperPoint::runNmsDilate(GLuint maskTexture, GLuint suppressionTexture)
{
    nmsDilateShader->use();

    glUniform1i(nmsDilateRadiusLocation, config.nmsRadius);

    bindImage(0, maskTexture, GL_READ_ONLY, GL_R32F, false);
    bindImage(1, suppressionTexture, GL_WRITE_ONLY, GL_R32F, false);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);

    imageBarrier();
}


void SuperPoint::runNmsUpdate(GLuint currentMask, GLuint suppressionTexture, GLuint nextMask)
{
    nmsUpdateShader->use();

    glUniform1i(nmsUpdateRadiusLocation, config.nmsRadius);

    bindImage(0, heatmapTexture->getTextureID(), GL_READ_ONLY, GL_R32F, false);
    bindImage(1, currentMask, GL_READ_ONLY, GL_R32F, false);
    bindImage(2, suppressionTexture, GL_READ_ONLY, GL_R32F, false);
    bindImage(3, nextMask, GL_WRITE_ONLY, GL_R32F, false);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);

    imageBarrier();
}


//-----------Feature Extraction---------//

bool SuperPoint::extract(const unsigned char* rgbaPixels, SuperPointFeatures& features)
{
    features = SuperPointFeatures();

    if (!ready) {
        std::cerr << "SuperPoint: extract() called before a successful initialize()." << std::endl;
        return false;
    }

    if (rgbaPixels == nullptr) {
        std::cerr << "SuperPoint: extract() received a null image." << std::endl;
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(imageWidth) * imageHeight;

    features.imageWidth = imageWidth;
    features.imageHeight = imageHeight;

    // Ping-pong textures change size during the network, so reset them every image
    pingA->resize(imageHeight, imageWidth, 64);
    pingB->resize(imageHeight, imageWidth, 64);

    const GLuint ping_a = pingA->getTextureID();
    const GLuint ping_b = pingB->getTextureID();


    // =====================================================================
    // Upload + Preprocessing
    // =====================================================================

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glBindTexture(GL_TEXTURE_2D, inputTexture->getTextureID());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight,
        GL_RGBA, GL_UNSIGNED_BYTE, rgbaPixels);
    glBindTexture(GL_TEXTURE_2D, 0);


    //-----------RGB2Gray---------//

    grayShader->use();

    bindImage(0, inputTexture->getTextureID(), GL_READ_ONLY, GL_RGBA8, false);
    bindImage(1, grayTexture->getTextureID(), GL_WRITE_ONLY, GL_RGBA8, false);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);
    imageBarrier();


    //-----------ToTensor---------//

    toTensorShader->use();

    bindImage(0, grayTexture->getTextureID(), GL_READ_ONLY, GL_RGBA8, false);
    bindImage(1, tensorTexture->getTextureID(), GL_WRITE_ONLY, GL_R32F, false);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);
    imageBarrier();


    // =====================================================================
    // Backbone - Block 1 (full resolution)
    // =====================================================================

    //-----------Conv1a + ReLU (single input channel)---------//

    conv1aShader->use();

    bindImage(0, tensorTexture->getTextureID(), GL_READ_ONLY, GL_R32F, false);
    bindImage(1, ping_a, GL_WRITE_ONLY, GL_R32F, true);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[CONV1A]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[CONV1A]);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 64);
    imageBarrier();

    runConv(CONV1B, ping_a, ping_b, true, imageWidth, imageHeight);

    pingA->resize(height1, width1, 64);
    runMaxPool(ping_b, ping_a, width1, height1, 64);


    // =====================================================================
    // Backbone - Block 2 (1/2)
    // =====================================================================

    pingB->resize(height1, width1, 64);
    runConv(CONV2A, ping_a, ping_b, true, width1, height1);
    runConv(CONV2B, ping_b, ping_a, true, width1, height1);

    pingB->resize(height2, width2, 64);
    runMaxPool(ping_a, ping_b, width2, height2, 64);


    // =====================================================================
    // Backbone - Block 3 (1/4)
    // =====================================================================

    pingA->resize(height2, width2, 128);
    runConv(CONV3A, ping_b, ping_a, true, width2, height2);

    pingB->resize(height2, width2, 128);
    runConv(CONV3B, ping_a, ping_b, true, width2, height2);

    pingA->resize(height3, width3, 128);
    runMaxPool(ping_b, ping_a, width3, height3, 128);


    // =====================================================================
    // Backbone - Block 4 (1/8)
    // =====================================================================

    pingB->resize(height3, width3, 128);
    runConv(CONV4A, ping_a, ping_b, true, width3, height3);
    runConv(CONV4B, ping_b, ping_a, true, width3, height3);


    // =====================================================================
    // Descriptor Head
    // =====================================================================

    pingB->resize(height3, width3, 256);
    runConv(CONVDA, ping_a, ping_b, true, width3, height3);
    runConv(CONVDB, ping_b, descriptorTexture->getTextureID(), false, width3, height3);


    // =====================================================================
    // Detector Head
    // =====================================================================

    runConv(CONVPA, ping_a, ping_b, true, width3, height3);
    runConv(CONVPB, ping_b, detectorTexture->getTextureID(), false, width3, height3);


    // =====================================================================
    // Descriptor L2 Normalization (result stays in pingB)
    // =====================================================================

    descriptorL2Shader->use();

    glUniform1f(epsilonLocation, 1e-12f);

    bindImage(0, descriptorTexture->getTextureID(), GL_READ_ONLY, GL_R32F, true);
    bindImage(1, ping_b, GL_WRITE_ONLY, GL_R32F, true);

    glDispatchCompute(groups(width3), groups(height3), 1);
    imageBarrier();


    // =====================================================================
    // Detector Softmax + Heatmap
    // =====================================================================

    softmaxShader->use();

    bindImage(0, detectorTexture->getTextureID(), GL_READ_ONLY, GL_R32F, true);
    bindImage(1, heatmapTexture->getTextureID(), GL_WRITE_ONLY, GL_R32F, false);

    glDispatchCompute(groups(width3), groups(height3), 1);
    imageBarrier();


    // =====================================================================
    // Non-Maximum Suppression
    // =====================================================================

    const GLuint maskA = nmsMaskA->getTextureID();
    const GLuint maskB = nmsMaskB->getTextureID();
    const GLuint suppression = nmsSuppression->getTextureID();

    nmsInitShader->use();

    glUniform1i(nmsInitRadiusLocation, config.nmsRadius);

    bindImage(0, heatmapTexture->getTextureID(), GL_READ_ONLY, GL_R32F, false);
    bindImage(1, maskA, GL_WRITE_ONLY, GL_R32F, false);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);
    imageBarrier();

    // Iteration 1 : A -> B
    runNmsDilate(maskA, suppression);
    runNmsUpdate(maskA, suppression, maskB);

    // Iteration 2 : B -> A (final mask ends in A)
    runNmsDilate(maskB, suppression);
    runNmsUpdate(maskB, suppression, maskA);


    // =====================================================================
    // Threshold + Border
    // =====================================================================

    thresholdShader->use();

    glUniform1f(thresholdLocation, config.detectionThreshold);
    glUniform1i(borderLocation, config.borderSize);

    bindImage(0, heatmapTexture->getTextureID(), GL_READ_ONLY, GL_R32F, false);
    bindImage(1, maskA, GL_READ_ONLY, GL_R32F, false);
    bindImage(2, filteredScoreTexture->getTextureID(), GL_WRITE_ONLY, GL_R32F, false);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);
    imageBarrier();


    // =====================================================================
    // Keypoint Compaction
    // =====================================================================

    const GLuint zero = 0;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointCounterBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);

    compactShader->use();

    glUniform1ui(keypointCapacityLocation, static_cast<GLuint>(pixelCount));

    bindImage(0, filteredScoreTexture->getTextureID(), GL_READ_ONLY, GL_R32F, false);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, keypointCounterBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, keypointCandidateBuffer);

    glDispatchCompute(groups(imageWidth), groups(imageHeight), 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);


    //-----------Read Back Count + Candidates---------//

    GLuint detectedCount = 0;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointCounterBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &detectedCount);

    if (detectedCount > pixelCount) {
        detectedCount = static_cast<GLuint>(pixelCount);
    }

    if (config.verbose) {
        std::cout << "Detected keypoints before top-K: " << detectedCount << std::endl;
    }

    std::vector<GLuint> candidateData(static_cast<size_t>(detectedCount) * 2);

    if (detectedCount > 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointCandidateBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
            static_cast<GLsizeiptr>(candidateData.size() * sizeof(GLuint)),
            candidateData.data());
    }


    //-----------Decode, Sort by Score, Keep Top-K---------//

    std::vector<std::tuple<float, GLuint, GLuint>> candidates;
    candidates.reserve(detectedCount);

    for (GLuint k = 0; k < detectedCount; k++) {

        const GLuint linearIndex = candidateData[k * 2];

        float score = 0.0f;
        std::memcpy(&score, &candidateData[k * 2 + 1], sizeof(float));

        candidates.emplace_back(
            score,
            linearIndex % static_cast<GLuint>(imageWidth),
            linearIndex / static_cast<GLuint>(imageWidth));
    }

    std::sort(candidates.begin(), candidates.end(),
        std::greater<std::tuple<float, GLuint, GLuint>>());

    if (candidates.size() > config.maxKeypoints) {
        candidates.resize(config.maxKeypoints);
    }

    const GLuint selectedCount = static_cast<GLuint>(candidates.size());

    if (config.verbose) {
        std::cout << "Selected keypoints after top-K: " << selectedCount << std::endl;
    }

    features.keypoints.reserve(selectedCount);

    for (const auto& c : candidates) {
        Keypoint kp;
        kp.score = std::get<0>(c);
        kp.x = static_cast<float>(std::get<1>(c));
        kp.y = static_cast<float>(std::get<2>(c));
        features.keypoints.push_back(kp);
    }

    if (selectedCount == 0) {
        return true;
    }


    // =====================================================================
    // Bilinear Descriptor Sampling
    // =====================================================================

    // Overwrite the candidate buffer with the selected (x, y) pairs
    std::vector<GLuint> coordinates(static_cast<size_t>(selectedCount) * 2);

    for (GLuint k = 0; k < selectedCount; k++) {
        coordinates[k * 2] = std::get<1>(candidates[k]);
        coordinates[k * 2 + 1] = std::get<2>(candidates[k]);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointCandidateBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
        static_cast<GLsizeiptr>(coordinates.size() * sizeof(GLuint)),
        coordinates.data());

    sampleShader->use();

    glUniform1ui(sampleKeypointCountLocation, selectedCount);

    bindImage(0, ping_b, GL_READ_ONLY, GL_R32F, true);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, keypointCandidateBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sampledDescriptorBuffer);

    glDispatchCompute(groups(static_cast<int>(selectedCount), kSampleLocalSize), 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    features.descriptors.resize(static_cast<size_t>(selectedCount) * kDescriptorDimension);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sampledDescriptorBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
        static_cast<GLsizeiptr>(features.descriptors.size() * sizeof(float)),
        features.descriptors.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return true;
}


//-----------Saving---------//

bool SuperPoint::save(const std::string& path, const SuperPointFeatures& features)
{
    std::ofstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "SuperPoint: could not create feature file '" << path << "'" << std::endl;
        return false;
    }

    const uint32_t magic = 0x53505054;
    const uint32_t version = 1;
    const uint32_t numberOfKeypoints = static_cast<uint32_t>(features.keypoints.size());
    const uint32_t descriptorSize = SuperPointFeatures::DescriptorDimension;

    file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&numberOfKeypoints), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&descriptorSize), sizeof(uint32_t));

    for (const Keypoint& kp : features.keypoints) {
        file.write(reinterpret_cast<const char*>(&kp.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&kp.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&kp.score), sizeof(float));
    }

    if (!features.descriptors.empty()) {
        file.write(reinterpret_cast<const char*>(features.descriptors.data()),
            static_cast<std::streamsize>(features.descriptors.size() * sizeof(float)));
    }

    file.close();

    return static_cast<bool>(file);
}