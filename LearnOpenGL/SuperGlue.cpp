#include "SuperGlue.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>


namespace
{
    //-----------Network Description (must match tools/export_superglue_weights.py)---------//

    constexpr int kDescriptorDimension = 256;

    constexpr int kEncoderLayerCount = 5;
    constexpr int kGnnLayerCount = 18;

    // Convolution index map
    constexpr int kEncoderBase = 0;                     // 5 convolutions
    constexpr int kGnnBase = kEncoderLayerCount;        // 6 convolutions per GNN layer
    constexpr int kGnnStride = 6;
    constexpr int kFinalProj = kGnnBase + kGnnLayerCount * kGnnStride;
    constexpr int kConvCount = kFinalProj + 1;

    // Offsets inside one GNN layer
    constexpr int kProjQuery = 0;
    constexpr int kProjKey = 1;
    constexpr int kProjValue = 2;
    constexpr int kMerge = 3;
    constexpr int kMlp0 = 4;
    constexpr int kMlp1 = 5;

    constexpr int kAttentionLocalSize = 64;
    constexpr int kAttentionHeads = 4;
    constexpr int kAttentionSharedBytes = 2 * 64 * 32 * sizeof(float);

    constexpr int kConvTile = 16;
    constexpr int kColumnLocalSize = 64;

    struct LayerShape
    {
        int cin;
        int cout;
    };

    std::vector<LayerShape> buildLayerShapes()
    {
        std::vector<LayerShape> shapes;

        shapes.push_back({ 3, 32 });
        shapes.push_back({ 32, 64 });
        shapes.push_back({ 64, 128 });
        shapes.push_back({ 128, 256 });
        shapes.push_back({ 256, 256 });

        for (int l = 0; l < kGnnLayerCount; l++) {
            shapes.push_back({ 256, 256 });   // query projection
            shapes.push_back({ 256, 256 });   // key projection
            shapes.push_back({ 256, 256 });   // value projection
            shapes.push_back({ 256, 256 });   // merge
            shapes.push_back({ 512, 512 });   // mlp[0]
            shapes.push_back({ 512, 256 });   // mlp[1]
        }

        shapes.push_back({ 256, 256 });       // final projection

        return shapes;
    }

    const std::vector<LayerShape>& layerShapes()
    {
        static const std::vector<LayerShape> shapes = buildLayerShapes();
        return shapes;
    }

    inline GLuint groups(int n, int localSize)
    {
        return static_cast<GLuint>((n + localSize - 1) / localSize);
    }

    inline void storageBarrier()
    {
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}


//-----------Construction / Destruction---------//

SuperGlue::SuperGlue(const SuperGlueConfig& cfg)
    : config(cfg)
{
}


SuperGlue::~SuperGlue()
{
    releaseResources();
}


void SuperGlue::releaseResources()
{
    ready = false;

    convShader.reset();
    attentionShader.reset();
    scoresShader.reset();
    sinkhornRowsShader.reset();
    sinkhornColsShader.reset();
    matchRowsShader.reset();
    matchColsShader.reset();

    if (!weightBuffers.empty()) {
        glDeleteBuffers(static_cast<GLsizei>(weightBuffers.size()), weightBuffers.data());
        weightBuffers.clear();
    }

    if (!biasBuffers.empty()) {
        glDeleteBuffers(static_cast<GLsizei>(biasBuffers.size()), biasBuffers.data());
        biasBuffers.clear();
    }

    if (!allBuffers.empty()) {
        glDeleteBuffers(static_cast<GLsizei>(allBuffers.size()), allBuffers.data());
        allBuffers.clear();
    }

    std::fill(std::begin(keypointInput), std::end(keypointInput), 0u);
    std::fill(&descriptors[0][0], &descriptors[0][0] + 4, 0u);
    std::fill(std::begin(scratch), std::end(scratch), 0u);

    queryBuffer = keyBuffer = valueBuffer = 0;
    messageBuffer = mergedBuffer = hiddenBuffer = 0;
    couplingBuffer = uBuffer = vBuffer = 0;
    indices0Buffer = maxScores0Buffer = indices1Buffer = maxScores1Buffer = 0;

    capacity = 0;
}


//-----------Initialization---------//

bool SuperGlue::initialize()
{
    releaseResources();

    if (config.maxKeypoints == 0 || config.sinkhornIterations <= 0) {
        std::cerr << "SuperGlue: maxKeypoints and sinkhornIterations must be positive." << std::endl;
        return false;
    }

    capacity = config.maxKeypoints;


    //-----------Check OpenGL Limits---------//

    GLint sharedMemory = 0;
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &sharedMemory);

    if (sharedMemory < kAttentionSharedBytes) {
        std::cerr << "SuperGlue: the attention shader needs " << kAttentionSharedBytes
            << " bytes of compute shared memory, this GPU offers " << sharedMemory << "." << std::endl;
        return false;
    }

    GLint64 maxSSBOSize = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxSSBOSize);

    const size_t couplingBytes = (capacity + 1) * (capacity + 1) * sizeof(float);
    const size_t hiddenBytes = static_cast<size_t>(2 * kDescriptorDimension) * capacity * sizeof(float);

    if (couplingBytes > static_cast<size_t>(maxSSBOSize) ||
        hiddenBytes > static_cast<size_t>(maxSSBOSize)) {

        std::cerr << "SuperGlue: required SSBO size exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE ("
            << maxSSBOSize / (1024.0 * 1024.0) << " MB). Lower maxKeypoints." << std::endl;
        return false;
    }


    //-----------Create Resources---------//

    if (!createShaders() ||
        !queryUniforms() ||
        !loadWeights() ||
        !createBuffers()) {

        releaseResources();
        return false;
    }

    ready = true;

    return true;
}


bool SuperGlue::createShaders()
{
    const std::filesystem::path dir(config.shaderDirectory);

    auto load = [&](const char* fileName) -> std::unique_ptr<ComputeShader>
        {
            auto shader = std::make_unique<ComputeShader>((dir / fileName).string());

            if (shader->getProgram() == 0) {
                std::cerr << "SuperGlue: failed to build shader '" << fileName << "'" << std::endl;
                return nullptr;
            }

            return shader;
        };

    convShader = load("SG_Conv1d.comp.txt");
    attentionShader = load("SG_Attention.comp.txt");
    scoresShader = load("SG_Scores.comp.txt");
    sinkhornRowsShader = load("SG_SinkhornRows.comp.txt");
    sinkhornColsShader = load("SG_SinkhornCols.comp.txt");
    matchRowsShader = load("SG_MatchRows.comp.txt");
    matchColsShader = load("SG_MatchCols.comp.txt");

    return convShader && attentionShader && scoresShader &&
        sinkhornRowsShader && sinkhornColsShader &&
        matchRowsShader && matchColsShader;
}


bool SuperGlue::queryUniforms()
{
    const GLuint conv = convShader->getProgram();
    const GLuint attn = attentionShader->getProgram();
    const GLuint scores = scoresShader->getProgram();
    const GLuint rows = sinkhornRowsShader->getProgram();
    const GLuint cols = sinkhornColsShader->getProgram();
    const GLuint matchRows = matchRowsShader->getProgram();
    const GLuint matchCols = matchColsShader->getProgram();

    uniforms.cin = glGetUniformLocation(conv, "uCin");
    uniforms.cout = glGetUniformLocation(conv, "uCout");
    uniforms.n = glGetUniformLocation(conv, "uN");
    uniforms.split = glGetUniformLocation(conv, "uSplit");
    uniforms.relu = glGetUniformLocation(conv, "uRelu");
    uniforms.addResidual = glGetUniformLocation(conv, "uAddResidual");

    uniforms.attnNq = glGetUniformLocation(attn, "uNq");
    uniforms.attnNs = glGetUniformLocation(attn, "uNs");

    uniforms.scoresM = glGetUniformLocation(scores, "uM");
    uniforms.scoresN = glGetUniformLocation(scores, "uN");
    uniforms.scoresDim = glGetUniformLocation(scores, "uDim");
    uniforms.scoresScale = glGetUniformLocation(scores, "uScale");
    uniforms.scoresAlpha = glGetUniformLocation(scores, "uAlpha");

    uniforms.rowsM = glGetUniformLocation(rows, "uM");
    uniforms.rowsN = glGetUniformLocation(rows, "uN");

    uniforms.colsM = glGetUniformLocation(cols, "uM");
    uniforms.colsN = glGetUniformLocation(cols, "uN");

    uniforms.matchRowsM = glGetUniformLocation(matchRows, "uM");
    uniforms.matchRowsN = glGetUniformLocation(matchRows, "uN");

    uniforms.matchColsM = glGetUniformLocation(matchCols, "uM");
    uniforms.matchColsN = glGetUniformLocation(matchCols, "uN");

    const GLint locations[] = {
        uniforms.cin, uniforms.cout, uniforms.n, uniforms.split, uniforms.relu, uniforms.addResidual,
        uniforms.attnNq, uniforms.attnNs,
        uniforms.scoresM, uniforms.scoresN, uniforms.scoresDim, uniforms.scoresScale, uniforms.scoresAlpha,
        uniforms.rowsM, uniforms.rowsN,
        uniforms.colsM, uniforms.colsN,
        uniforms.matchRowsM, uniforms.matchRowsN,
        uniforms.matchColsM, uniforms.matchColsN
    };

    for (GLint location : locations) {
        if (location == -1) {
            std::cerr << "SuperGlue: one or more compute shader uniforms were not found." << std::endl;
            return false;
        }
    }

    return true;
}


bool SuperGlue::loadWeights()
{
    const std::vector<LayerShape>& shapes = layerShapes();

    size_t expectedFloats = 1;   // bin_score

    for (const LayerShape& s : shapes) {
        expectedFloats += static_cast<size_t>(s.cout) * s.cin + s.cout;
    }

    std::ifstream file(config.weightsPath, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "SuperGlue: failed to open weights file '" << config.weightsPath << "'" << std::endl;
        return false;
    }

    const std::streamoff fileBytes = file.tellg();

    if (fileBytes != static_cast<std::streamoff>(expectedFloats * sizeof(float))) {
        std::cerr << "SuperGlue: weights file '" << config.weightsPath << "' has " << fileBytes
            << " bytes, expected " << expectedFloats * sizeof(float)
            << ". Re-export it with tools/export_superglue_weights.py." << std::endl;
        return false;
    }

    file.seekg(0, std::ios::beg);

    weightBuffers.assign(kConvCount, 0);
    biasBuffers.assign(kConvCount, 0);

    glGenBuffers(kConvCount, weightBuffers.data());
    glGenBuffers(kConvCount, biasBuffers.data());

    for (int l = 0; l < kConvCount; l++) {

        const size_t weightCount = static_cast<size_t>(shapes[l].cout) * shapes[l].cin;
        const size_t biasCount = static_cast<size_t>(shapes[l].cout);

        std::vector<float> w(weightCount);
        std::vector<float> b(biasCount);

        file.read(reinterpret_cast<char*>(w.data()), weightCount * sizeof(float));
        file.read(reinterpret_cast<char*>(b.data()), biasCount * sizeof(float));

        if (!file) {
            std::cerr << "SuperGlue: weights file is too short at convolution " << l << std::endl;
            return false;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightBuffers[l]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, weightCount * sizeof(float), w.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, biasBuffers[l]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, biasCount * sizeof(float), b.data(), GL_STATIC_DRAW);
    }

    file.read(reinterpret_cast<char*>(&binScore), sizeof(float));

    if (!file) {
        std::cerr << "SuperGlue: weights file is missing the dustbin score." << std::endl;
        return false;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    if (config.verbose) {
        std::cout << "SuperGlue: loaded " << expectedFloats << " parameters from '"
            << config.weightsPath << "' (dustbin score " << binScore << ")" << std::endl;
    }

    return true;
}


GLuint SuperGlue::allocateBuffer(size_t floatCount)
{
    GLuint buffer = 0;

    glGenBuffers(1, &buffer);

    if (buffer == 0) {
        return 0;
    }

    allBuffers.push_back(buffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(floatCount * sizeof(float)), NULL, GL_DYNAMIC_COPY);

    return buffer;
}


bool SuperGlue::createBuffers()
{
    const size_t n = capacity;
    const size_t d = kDescriptorDimension;

    for (int i = 0; i < 2; i++) {
        keypointInput[i] = allocateBuffer(3 * n);
        descriptors[i][0] = allocateBuffer(d * n);
        descriptors[i][1] = allocateBuffer(d * n);
    }

    scratch[0] = allocateBuffer(d * n);
    scratch[1] = allocateBuffer(d * n);

    queryBuffer = allocateBuffer(d * n);
    keyBuffer = allocateBuffer(d * n);
    valueBuffer = allocateBuffer(d * n);
    messageBuffer = allocateBuffer(d * n);
    mergedBuffer = allocateBuffer(d * n);
    hiddenBuffer = allocateBuffer(2 * d * n);

    couplingBuffer = allocateBuffer((n + 1) * (n + 1));
    uBuffer = allocateBuffer(n + 1);
    vBuffer = allocateBuffer(n + 1);

    indices0Buffer = allocateBuffer(n);
    maxScores0Buffer = allocateBuffer(n);
    indices1Buffer = allocateBuffer(n);
    maxScores1Buffer = allocateBuffer(n);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for (GLuint buffer : allBuffers) {
        if (buffer == 0) {
            std::cerr << "SuperGlue: failed to generate working buffers." << std::endl;
            return false;
        }
    }

    const GLenum error = glGetError();

    if (error == GL_OUT_OF_MEMORY) {
        std::cerr << "SuperGlue: out of GPU memory while allocating buffers. Lower maxKeypoints." << std::endl;
        return false;
    }

    return true;
}


//-----------Layer Helpers---------//

// out[co][n] = bias + W * in   (+ ReLU) (+ residual)
// With inputB the input is the channel concatenation [inputA ; inputB], inputA holding `split` channels.
void SuperGlue::runConv(int layer, GLuint inputA, GLuint output, int n,
    bool relu, GLuint inputB, int split, GLuint residual)
{
    const LayerShape& shape = layerShapes()[layer];

    convShader->use();

    glUniform1i(uniforms.cin, shape.cin);
    glUniform1i(uniforms.cout, shape.cout);
    glUniform1i(uniforms.n, n);
    glUniform1i(uniforms.split, inputB != 0 ? split : shape.cin);
    glUniform1i(uniforms.relu, relu ? 1 : 0);
    glUniform1i(uniforms.addResidual, residual != 0 ? 1 : 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightBuffers[layer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasBuffers[layer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, inputA);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, inputB != 0 ? inputB : inputA);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, output);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, residual != 0 ? residual : inputA);

    glDispatchCompute(groups(n, kConvTile), groups(shape.cout, kConvTile), 1);

    storageBarrier();
}


void SuperGlue::runAttention(GLuint query, GLuint key, GLuint value, GLuint output, int nq, int ns)
{
    attentionShader->use();

    glUniform1i(uniforms.attnNq, nq);
    glUniform1i(uniforms.attnNs, ns);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, query);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, key);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, value);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, output);

    glDispatchCompute(groups(nq, kAttentionLocalSize), kAttentionHeads, 1);

    storageBarrier();
}


// One AttentionalPropagation layer:   output = x + mlp([x ; attention(x, source, source)])
void SuperGlue::runPropagation(int gnnLayer, GLuint x, GLuint source, GLuint output, int n, int ns)
{
    const int base = kGnnBase + gnnLayer * kGnnStride;

    runConv(base + kProjQuery, x, queryBuffer, n);
    runConv(base + kProjKey, source, keyBuffer, ns);
    runConv(base + kProjValue, source, valueBuffer, ns);

    runAttention(queryBuffer, keyBuffer, valueBuffer, messageBuffer, n, ns);

    runConv(base + kMerge, messageBuffer, mergedBuffer, n);

    runConv(base + kMlp0, x, hiddenBuffer, n, true, mergedBuffer, kDescriptorDimension);

    runConv(base + kMlp1, hiddenBuffer, output, n, false, 0, 0, x);
}


//-----------Input Upload---------//

void SuperGlue::uploadImage(int image, const SuperPointFeatures& features, int count)
{
    const size_t n = static_cast<size_t>(count);

    //-----------Normalized Keypoints + Scores (3 x N)---------//

    const float width = static_cast<float>(features.imageWidth);
    const float height = static_cast<float>(features.imageHeight);

    const float centerX = width * 0.5f;
    const float centerY = height * 0.5f;
    const float scaling = std::max(width, height) * 0.7f;

    std::vector<float> keypointData(3 * n);

    for (size_t k = 0; k < n; k++) {

        const Keypoint& kp = features.keypoints[k];

        keypointData[0 * n + k] = (kp.x - centerX) / scaling;
        keypointData[1 * n + k] = (kp.y - centerY) / scaling;
        keypointData[2 * n + k] = kp.score;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, keypointInput[image]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
        static_cast<GLsizeiptr>(keypointData.size() * sizeof(float)), keypointData.data());


    //-----------Descriptors: keypoint-major -> channel-major (256 x N)---------//

    std::vector<float> descriptorData(static_cast<size_t>(kDescriptorDimension) * n);

    for (size_t k = 0; k < n; k++) {
        for (size_t c = 0; c < kDescriptorDimension; c++) {
            descriptorData[c * n + k] = features.descriptors[k * kDescriptorDimension + c];
        }
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, descriptors[image][0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
        static_cast<GLsizeiptr>(descriptorData.size() * sizeof(float)), descriptorData.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}


//-----------Matching---------//

bool SuperGlue::match(const SuperPointFeatures& features0,
    const SuperPointFeatures& features1,
    SuperGlueMatches& result)
{
    result = SuperGlueMatches();

    if (!ready) {
        std::cerr << "SuperGlue: match() called before a successful initialize()." << std::endl;
        return false;
    }

    const SuperPointFeatures* inputs[2] = { &features0, &features1 };

    int counts[2] = {};

    for (int i = 0; i < 2; i++) {

        const SuperPointFeatures& f = *inputs[i];

        if (f.imageWidth <= 0 || f.imageHeight <= 0) {
            std::cerr << "SuperGlue: features of image " << i << " have no image size." << std::endl;
            return false;
        }

        counts[i] = static_cast<int>(std::min<size_t>(f.keypoints.size(), config.maxKeypoints));

        if (f.descriptors.size() < static_cast<size_t>(counts[i]) * kDescriptorDimension) {
            std::cerr << "SuperGlue: features of image " << i
                << " do not hold 256 descriptor values per keypoint." << std::endl;
            return false;
        }
    }

    const int m = counts[0];
    const int n = counts[1];

    result.keypointCount0 = m;
    result.keypointCount1 = n;

    result.matches0.assign(m, -1);
    result.matches1.assign(n, -1);
    result.scores0.assign(m, 0.0f);
    result.scores1.assign(n, 0.0f);

    // Nothing to match (the reference implementation returns -1 everywhere)
    if (m == 0 || n == 0) {
        return true;
    }

    const auto startTime = std::chrono::steady_clock::now();

    const int imageCount[2] = { m, n };


    // =====================================================================
    // Upload
    // =====================================================================

    uploadImage(0, features0, m);
    uploadImage(1, features1, n);


    // =====================================================================
    // Keypoint Encoder :  desc = desc + MLP(keypoints, scores)
    // =====================================================================

    for (int i = 0; i < 2; i++) {

        const int count = imageCount[i];

        runConv(kEncoderBase + 0, keypointInput[i], scratch[0], count, true);
        runConv(kEncoderBase + 1, scratch[0], scratch[1], count, true);
        runConv(kEncoderBase + 2, scratch[1], scratch[0], count, true);
        runConv(kEncoderBase + 3, scratch[0], scratch[1], count, true);

        runConv(kEncoderBase + 4, scratch[1], descriptors[i][1], count,
            false, 0, 0, descriptors[i][0]);
    }

    int current = 1;   // index of the ping-pong buffer holding the current descriptors


    // =====================================================================
    // Attentional Graph Neural Network : 9 x (self, cross)
    // =====================================================================

    for (int layer = 0; layer < kGnnLayerCount; layer++) {

        const bool cross = (layer % 2) == 1;

        // Both images read the descriptors of the previous layer and write to the other
        // ping-pong buffer, so the order of the two images does not matter.
        for (int i = 0; i < 2; i++) {

            const int sourceImage = cross ? (1 - i) : i;

            runPropagation(
                layer,
                descriptors[i][current],
                descriptors[sourceImage][current],
                descriptors[i][1 - current],
                imageCount[i],
                imageCount[sourceImage]
            );
        }

        current = 1 - current;
    }


    // =====================================================================
    // Final projection (result goes to the buffer that is not "current")
    // =====================================================================

    for (int i = 0; i < 2; i++) {
        runConv(kFinalProj, descriptors[i][current], descriptors[i][1 - current], imageCount[i]);
    }

    const GLuint matchingDescriptors0 = descriptors[0][1 - current];
    const GLuint matchingDescriptors1 = descriptors[1][1 - current];


    // =====================================================================
    // Optimal Transport (log-space Sinkhorn)
    // =====================================================================

    //-----------Coupling Matrix [(M + 1) x (N + 1)]---------//

    scoresShader->use();

    glUniform1i(uniforms.scoresM, m);
    glUniform1i(uniforms.scoresN, n);
    glUniform1i(uniforms.scoresDim, kDescriptorDimension);
    glUniform1f(uniforms.scoresScale, 1.0f / std::sqrt(static_cast<float>(kDescriptorDimension)));
    glUniform1f(uniforms.scoresAlpha, binScore);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matchingDescriptors0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, matchingDescriptors1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, couplingBuffer);

    glDispatchCompute(groups(n + 1, 16), groups(m + 1, 16), 1);

    storageBarrier();


    //-----------Reset Dual Variables---------//

    {
        const std::vector<float> zeros(static_cast<size_t>(std::max(m, n)) + 1, 0.0f);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, uBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
            static_cast<GLsizeiptr>((static_cast<size_t>(m) + 1) * sizeof(float)), zeros.data());

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
            static_cast<GLsizeiptr>((static_cast<size_t>(n) + 1) * sizeof(float)), zeros.data());

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }


    //-----------Sinkhorn Iterations---------//

    // Coupling / u / v use the same bindings (0, 1, 2) in every shader below
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, couplingBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, uBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vBuffer);

    sinkhornRowsShader->use();
    glUniform1i(uniforms.rowsM, m);
    glUniform1i(uniforms.rowsN, n);

    sinkhornColsShader->use();
    glUniform1i(uniforms.colsM, m);
    glUniform1i(uniforms.colsN, n);

    for (int it = 0; it < config.sinkhornIterations; it++) {

        sinkhornRowsShader->use();
        glDispatchCompute(static_cast<GLuint>(m + 1), 1, 1);
        storageBarrier();

        sinkhornColsShader->use();
        glDispatchCompute(groups(n + 1, kColumnLocalSize), 1, 1);
        storageBarrier();
    }


    // =====================================================================
    // Row / Column arg-max of the log assignment matrix
    // =====================================================================

    matchRowsShader->use();

    glUniform1i(uniforms.matchRowsM, m);
    glUniform1i(uniforms.matchRowsN, n);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, indices0Buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, maxScores0Buffer);

    glDispatchCompute(static_cast<GLuint>(m), 1, 1);

    matchColsShader->use();

    glUniform1i(uniforms.matchColsM, m);
    glUniform1i(uniforms.matchColsN, n);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, indices1Buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, maxScores1Buffer);

    glDispatchCompute(groups(n, kColumnLocalSize), 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);


    //-----------Read Back---------//

    std::vector<int> argmax0(m), argmax1(n);
    std::vector<float> logMax0(m), logMax1(n);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indices0Buffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m * sizeof(int), argmax0.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxScores0Buffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m * sizeof(float), logMax0.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indices1Buffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(int), argmax1.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxScores1Buffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(float), logMax1.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // =====================================================================
    // Mutual check + threshold (identical to the reference implementation)
    // =====================================================================

    std::vector<char> mutual0(m, 0), mutual1(n, 0);

    for (int i = 0; i < m; i++) {
        const int j = argmax0[i];
        mutual0[i] = (j >= 0 && j < n && argmax1[j] == i) ? 1 : 0;
    }

    for (int j = 0; j < n; j++) {
        const int i = argmax1[j];
        mutual1[j] = (i >= 0 && i < m && argmax0[i] == j) ? 1 : 0;
    }

    for (int i = 0; i < m; i++) {
        result.scores0[i] = mutual0[i] ? std::exp(logMax0[i]) : 0.0f;
    }

    for (int j = 0; j < n; j++) {
        result.scores1[j] = mutual1[j] ? result.scores0[argmax1[j]] : 0.0f;
    }

    std::vector<char> valid0(m, 0);

    for (int i = 0; i < m; i++) {
        valid0[i] = (mutual0[i] && result.scores0[i] > config.matchThreshold) ? 1 : 0;
        result.matches0[i] = valid0[i] ? argmax0[i] : -1;
    }

    for (int j = 0; j < n; j++) {
        const bool valid1 = mutual1[j] && valid0[argmax1[j]];
        result.matches1[j] = valid1 ? argmax1[j] : -1;
    }

    for (int i = 0; i < m; i++) {
        if (result.matches0[i] >= 0) {
            Match pair;
            pair.index0 = i;
            pair.index1 = result.matches0[i];
            pair.score = result.scores0[i];
            result.pairs.push_back(pair);
        }
    }

    if (config.verbose) {

        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - startTime).count();

        std::cout << "SuperGlue: " << m << " x " << n << " keypoints -> "
            << result.pairs.size() << " matches (" << milliseconds << " ms)" << std::endl;
    }

    return true;
}


//-----------Saving---------//

bool SuperGlue::save(const std::string& path, const SuperGlueMatches& matches)
{
    std::ofstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "SuperGlue: could not create match file '" << path << "'" << std::endl;
        return false;
    }

    const uint32_t magic = 0x53474D54;   // "SGMT"
    const uint32_t version = 1;
    const uint32_t count0 = static_cast<uint32_t>(matches.keypointCount0);
    const uint32_t count1 = static_cast<uint32_t>(matches.keypointCount1);
    const uint32_t matchCount = static_cast<uint32_t>(matches.pairs.size());

    file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&count0), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&count1), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&matchCount), sizeof(uint32_t));

    for (const Match& pair : matches.pairs) {

        const uint32_t index0 = static_cast<uint32_t>(pair.index0);
        const uint32_t index1 = static_cast<uint32_t>(pair.index1);

        file.write(reinterpret_cast<const char*>(&index0), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&index1), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&pair.score), sizeof(float));
    }

    file.close();

    return static_cast<bool>(file);
}