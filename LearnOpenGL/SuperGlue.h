#pragma once

#include <glad.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ComputeShader.h"
#include "SuperPoint.h"


//-----------Configuration---------//

struct SuperGlueConfig
{
    int sinkhornIterations = 100;

    float matchThreshold = 0.2f;

    // Only the first maxKeypoints keypoints of each image are matched.
    // SuperPoint returns keypoints sorted by descending score, so these are the best ones.
    // SuperGlue's cost grows with the square of this value (1024 is the usual choice).
    GLuint maxKeypoints = 1024;

    std::string shaderDirectory = "Resources/Shaders/";
    std::string weightsPath = "Resources/Weights/superglue_indoor.bin";   // or superglue_outdoor.bin

    bool verbose = true;
};


//-----------Output---------//

struct Match
{
    int index0 = -1;      // keypoint index in the first image
    int index1 = -1;      // keypoint index in the second image
    float score = 0.0f;   // matching confidence in [0, 1]
};

struct SuperGlueMatches
{
    // Number of keypoints that were actually used (after the maxKeypoints cap)
    int keypointCount0 = 0;
    int keypointCount1 = 0;

    // Same convention as the reference implementation: -1 means "no match"
    std::vector<int> matches0;        // keypointCount0 entries, index into image 1
    std::vector<int> matches1;        // keypointCount1 entries, index into image 0

    std::vector<float> scores0;       // keypointCount0 entries
    std::vector<float> scores1;       // keypointCount1 entries

    // Convenience list of the valid matches, sorted by index0
    std::vector<Match> pairs;

    size_t count() const { return pairs.size(); }
};


//-----------SuperGlue (OpenGL compute pipeline)---------//
//
// Matches two sets of SuperPoint features (keypoints + 256-D descriptors).
//
// Every method that touches OpenGL (constructor excluded, initialize, match and
// the destructor) must be called while the OpenGL context is current.
// Destroy the object BEFORE destroying the window / context.
//
// The weights file is produced by tools/export_superglue_weights.py.

class SuperGlue
{
public:

    explicit SuperGlue(const SuperGlueConfig& config = SuperGlueConfig());

    SuperGlue(const SuperGlue&) = delete;
    SuperGlue& operator=(const SuperGlue&) = delete;

    ~SuperGlue();


    // Compiles shaders, loads the weights and allocates every buffer.
    // Independent of the image size.
    bool initialize();

    bool isReady() const { return ready; }

    const SuperGlueConfig& getConfig() const { return config; }


    // Features must come from SuperPoint::extract (descriptors L2-normalized,
    // keypoints sorted by descending score).
    bool match(const SuperPointFeatures& features0,
        const SuperPointFeatures& features1,
        SuperGlueMatches& matches);


    // Binary format: [magic, version, keypointCount0, keypointCount1, matchCount] (uint32),
    // then matchCount x (index0 uint32, index1 uint32, score float).
    static bool save(const std::string& path, const SuperGlueMatches& matches);


private:

    struct Uniforms
    {
        GLint cin = -1, cout = -1, n = -1, split = -1, relu = -1, addResidual = -1;   // conv
        GLint attnNq = -1, attnNs = -1;                                               // attention
        GLint scoresM = -1, scoresN = -1, scoresDim = -1, scoresScale = -1, scoresAlpha = -1;
        GLint rowsM = -1, rowsN = -1;
        GLint colsM = -1, colsN = -1;
        GLint matchRowsM = -1, matchRowsN = -1;
        GLint matchColsM = -1, matchColsN = -1;
    };

    bool createShaders();
    bool queryUniforms();
    bool loadWeights();
    bool createBuffers();
    void releaseResources();

    GLuint allocateBuffer(size_t floatCount);

    void uploadImage(int image, const SuperPointFeatures& features, int count);

    void runConv(int layer, GLuint inputA, GLuint output, int n,
        bool relu = false, GLuint inputB = 0, int split = 0, GLuint residual = 0);

    void runAttention(GLuint query, GLuint key, GLuint value, GLuint output, int nq, int ns);

    void runPropagation(int gnnLayer, GLuint x, GLuint source, GLuint output, int n, int ns);


    SuperGlueConfig config;

    bool ready = false;

    // Shaders
    std::unique_ptr<ComputeShader> convShader;
    std::unique_ptr<ComputeShader> attentionShader;
    std::unique_ptr<ComputeShader> scoresShader;
    std::unique_ptr<ComputeShader> sinkhornRowsShader;
    std::unique_ptr<ComputeShader> sinkhornColsShader;
    std::unique_ptr<ComputeShader> matchRowsShader;
    std::unique_ptr<ComputeShader> matchColsShader;

    Uniforms uniforms;

    // Weights (one SSBO per convolution, BatchNorm already folded in)
    std::vector<GLuint> weightBuffers;
    std::vector<GLuint> biasBuffers;
    float binScore = 0.0f;

    // Working buffers (all [channels][keypoints], channel-major)
    std::vector<GLuint> allBuffers;

    GLuint keypointInput[2] = {};          // 3 x N : normalized x, y, score
    GLuint descriptors[2][2] = {};         // 256 x N, ping-pong per image
    GLuint scratch[2] = {};
    GLuint queryBuffer = 0;
    GLuint keyBuffer = 0;
    GLuint valueBuffer = 0;
    GLuint messageBuffer = 0;
    GLuint mergedBuffer = 0;
    GLuint hiddenBuffer = 0;               // 512 x N

    GLuint couplingBuffer = 0;             // (M + 1) x (N + 1)
    GLuint uBuffer = 0;
    GLuint vBuffer = 0;

    GLuint indices0Buffer = 0;
    GLuint maxScores0Buffer = 0;
    GLuint indices1Buffer = 0;
    GLuint maxScores1Buffer = 0;

    size_t capacity = 0;                   // maxKeypoints the buffers were sized for
};