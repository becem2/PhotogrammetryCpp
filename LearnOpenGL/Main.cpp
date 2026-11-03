
#define NOMINMAX
#define GLFW_INCLUDE_NONE

// Windows / DXGI
#include <Windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

// GLAD / GLFW / IMGUI
#include <glad.h>
#include <glfw3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// C++ STL
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <functional>
#include <system_error>

// stb_image
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// My Classes
#include "ComputeShader.h"
#include "Texture.h"


//-----------GPU Memory Information---------//

void CheckGPUMemory()
{
    IDXGIFactory1* factory = nullptr;

    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        return;
    }

    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* adapter = nullptr;

        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter->Release();
            continue;
        }

        IDXGIAdapter3* adapter3 = nullptr;

        if (FAILED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3)))) {
            adapter->Release();
            continue;
        }

        DXGI_QUERY_VIDEO_MEMORY_INFO info{};

        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {

            const double GB = 1024.0 * 1024.0 * 1024.0;

            double totalVRAM = desc.DedicatedVideoMemory / GB;
            double budget = info.Budget / GB;
            double usage = info.CurrentUsage / GB;

            double available = (info.Budget > info.CurrentUsage)
                ? (info.Budget - info.CurrentUsage) / GB
                : 0.0;

            std::wcout << L"\nGPU: " << desc.Description << L"\n";
            std::wcout << L"Total Dedicated VRAM: " << totalVRAM << L" GB\n";
            std::wcout << L"Application Memory Budget: " << budget << L" GB\n";
            std::wcout << L"Application Memory Usage: " << usage << L" GB\n";
            std::wcout << L"Remaining Budget: " << available << L" GB\n";
        }

        adapter3->Release();
        adapter->Release();
    }

    factory->Release();
}


//-----------Function Declarations---------//

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void processInput(GLFWwindow* window);
unsigned int loadTexture(const char* path);


// screen dimensions

int screenWidth = 0;
int screenHeight = 0;


// timing

float deltaTime = 0.0f;
float lastFrame = 0.0f;


// cursor capture state

enum class CursorMode { Captured, Free };

CursorMode cursorMode = CursorMode::Free;


// =====================================================================
// MAIN
// =====================================================================

int main()
{
    // glfw: initialize and configure
    // ------------------------------

    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif


    // glfw window creation
    // --------------------

    screenWidth = 800;
    screenHeight = 600;

    GLFWwindow* window = glfwCreateWindow(
        screenWidth,
        screenHeight,
        "LearnOpenGL",
        NULL,
        NULL
    );

    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);


    // glad: load all OpenGL function pointers
    // ---------------------------------------

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    // configure global OpenGL state
    // -----------------------------

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);


    //-----------Check VRAM---------//

    CheckGPUMemory();


    // =====================================================================
    // Image Loading
    // =====================================================================

    std::vector<std::string> imagePaths;

    std::string folderPath = "Resources/Images";

    std::filesystem::directory_entry folderEntry(folderPath);

    if (!folderEntry.exists() || !folderEntry.is_directory()) {
        std::cout << "ERROR : " << folderPath
            << " is not a directory or it doesn't exist" << std::endl;

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }

    int filesCounter = 0;

    for (const auto& imageEntry : std::filesystem::directory_iterator(folderPath)) {

        if (imageEntry.is_regular_file()) {

            std::string extension = imageEntry.path().extension().string();

            for (char& c : extension) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }

            if (extension == ".png" ||
                extension == ".jpg" ||
                extension == ".jpeg") {

                imagePaths.push_back(imageEntry.path().string());
            }

            filesCounter++;
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());

    const int imageCounter = static_cast<int>(imagePaths.size());

    std::cout << "Found " << filesCounter
        << " files in the '" << folderPath
        << "' folder " << imageCounter
        << " of them are images" << std::endl;

    for (int i = 0; i < imageCounter; i++) {
        std::cout << "Image number " << i
            << " : '" << imagePaths[i] << "'" << std::endl;
    }

    if (imageCounter == 0) {
        std::cout << "No images found!" << std::endl;

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    //-----------Load First Image---------//

    int imageWidth = 0;
    int imageHeight = 0;
    int imageChannels = 0;

    unsigned char* imageData = stbi_load(
        imagePaths[0].c_str(),
        &imageWidth,
        &imageHeight,
        &imageChannels,
        4
    );

    if (!imageData) {
        std::cout << "Failed to load first image!" << std::endl;

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }

    const int textureWidth = imageWidth;
    const int textureHeight = imageHeight;


    //-----------Validate Image Dimensions---------//

    if (textureWidth <= 0 ||
        textureHeight <= 0 ||
        textureWidth % 8 != 0 ||
        textureHeight % 8 != 0) {

        std::cout << "ERROR: Image dimensions must be positive and divisible by 8." << std::endl;

        stbi_image_free(imageData);

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    //-----------Check OpenGL Texture Limits---------//

    GLint maxTextureSize = 0;
    GLint maxArrayLayers = 0;

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxArrayLayers);

    if (textureWidth > maxTextureSize ||
        textureHeight > maxTextureSize ||
        maxArrayLayers < 256) {

        std::cout << "ERROR: Image dimensions or array layer count exceed OpenGL limits." << std::endl;

        std::cout << "Maximum texture size: " << maxTextureSize << std::endl;
        std::cout << "Maximum array layers: " << maxArrayLayers << std::endl;

        stbi_image_free(imageData);

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    //-----------Feature Map Dimensions---------//

    const int width1 = textureWidth / 2;
    const int height1 = textureHeight / 2;

    const int width2 = textureWidth / 4;
    const int height2 = textureHeight / 4;

    const int width3 = textureWidth / 8;
    const int height3 = textureHeight / 8;


    //-----------Keypoint Configuration---------//

    const int nmsRadius = 4;
    const int borderSize = 4;

    const float detectionThreshold = 0.005f;

    const GLuint maxKeypoints = 4096;

    const GLuint descriptorDimension = 256;


    //-----------Candidate Buffer Capacity---------//

    const size_t pixelCount =
        static_cast<size_t>(textureWidth) * textureHeight;

    const size_t candidateBufferSize =
        pixelCount * sizeof(GLuint) * 2;

    const size_t descriptorBufferSize =
        static_cast<size_t>(maxKeypoints) *
        descriptorDimension *
        sizeof(float);


    //-----------Check SSBO Limits---------//

    GLint64 maxSSBOSize = 0;

    glGetInteger64v(
        GL_MAX_SHADER_STORAGE_BLOCK_SIZE,
        &maxSSBOSize
    );

    if (candidateBufferSize > static_cast<size_t>(maxSSBOSize) ||
        descriptorBufferSize > static_cast<size_t>(maxSSBOSize)) {

        std::cout << "ERROR: Required SSBO allocation exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE." << std::endl;

        std::cout << "Candidate buffer size: "
            << candidateBufferSize / (1024.0 * 1024.0)
            << " MB" << std::endl;

        std::cout << "Maximum SSBO size: "
            << maxSSBOSize / (1024.0 * 1024.0)
            << " MB" << std::endl;

        stbi_image_free(imageData);

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    //-----------Create Feature Directory---------//

    std::error_code directoryError;

    std::filesystem::create_directories(
        "Resources/Features",
        directoryError
    );

    if (directoryError) {
        std::cout << "ERROR: Could not create Resources/Features directory: "
            << directoryError.message() << std::endl;

        stbi_image_free(imageData);

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    // =====================================================================
    // GPU Resources
    // =====================================================================

    {
        //-----------Input Textures---------//

        Texture inputTexture(
            LayerType::NORMAL,
            false,
            GL_RGBA8,
            textureHeight,
            textureWidth,
            0
        );

        Texture grayTexture(
            LayerType::NORMAL,
            false,
            GL_RGBA8,
            textureHeight,
            textureWidth,
            0
        );

        Texture tensorTexture(
            LayerType::NORMAL,
            false,
            GL_R32F,
            textureHeight,
            textureWidth,
            0
        );


        //-----------Ping-Pong Textures---------//

        Texture pingATexture(
            LayerType::NORMAL,
            true,
            GL_R32F,
            textureHeight,
            textureWidth,
            64
        );

        Texture pingBTexture(
            LayerType::NORMAL,
            true,
            GL_R32F,
            textureHeight,
            textureWidth,
            64
        );


        //-----------Descriptor Output---------//

        Texture descriptorTexture(
            LayerType::NORMAL,
            true,
            GL_R32F,
            height3,
            width3,
            256
        );


        //-----------Detector Output---------//

        Texture detectorTexture(
            LayerType::NORMAL,
            true,
            GL_R32F,
            height3,
            width3,
            65
        );


        //-----------Detector Heatmap---------//

        Texture heatmapTexture(
            LayerType::NORMAL,
            false,
            GL_R32F,
            textureHeight,
            textureWidth,
            0
        );


        //-----------NMS Textures---------//

        Texture nmsMaskATexture(
            LayerType::NORMAL,
            false,
            GL_R32F,
            textureHeight,
            textureWidth,
            0
        );

        Texture nmsMaskBTexture(
            LayerType::NORMAL,
            false,
            GL_R32F,
            textureHeight,
            textureWidth,
            0
        );

        Texture nmsSuppressionTexture(
            LayerType::NORMAL,
            false,
            GL_R32F,
            textureHeight,
            textureWidth,
            0
        );

        Texture filteredScoreTexture(
            LayerType::NORMAL,
            false,
            GL_R32F,
            textureHeight,
            textureWidth,
            0
        );


        //-----------Check Texture Creation---------//

        bool computeReady = true;

        if (inputTexture.getTextureID() == 0 ||
            grayTexture.getTextureID() == 0 ||
            tensorTexture.getTextureID() == 0 ||
            pingATexture.getTextureID() == 0 ||
            pingBTexture.getTextureID() == 0 ||
            descriptorTexture.getTextureID() == 0 ||
            detectorTexture.getTextureID() == 0 ||
            heatmapTexture.getTextureID() == 0 ||
            nmsMaskATexture.getTextureID() == 0 ||
            nmsMaskBTexture.getTextureID() == 0 ||
            nmsSuppressionTexture.getTextureID() == 0 ||
            filteredScoreTexture.getTextureID() == 0) {

            std::cout << "ERROR: Failed to create one or more textures!" << std::endl;

            computeReady = false;
        }


        //------------------------------------------------------------------
        // Shaders
        //------------------------------------------------------------------

        ComputeShader GrayShader(
            "Resources/Shaders/RGB2Gray.comp.txt"
        );

        ComputeShader ToTensorShader(
            "Resources/Shaders/ToTensor.comp.txt"
        );

        ComputeShader Conv1aReLUShader(
            "Resources/Shaders/Conv1aReLU.comp.txt"
        );

        ComputeShader ConvArrayReLUShader(
            "Resources/Shaders/ConvArrayReLU.comp.txt"
        );

        ComputeShader ConvArrayShader(
            "Resources/Shaders/ConvArray.comp.txt"
        );

        ComputeShader MaxPoolShader(
            "Resources/Shaders/MaxPool.comp.txt"
        );

        ComputeShader DescriptorL2Shader(
            "Resources/Shaders/DescriptorL2.comp.txt"
        );

        ComputeShader DetectorSoftmaxShader(
            "Resources/Shaders/DetectorSoftmaxHeatmap.comp.txt"
        );

        ComputeShader NMSInitShader(
            "Resources/Shaders/NMSInit.comp.txt"
        );

        ComputeShader NMSDilateShader(
            "Resources/Shaders/NMSDilate.comp.txt"
        );

        ComputeShader NMSUpdateShader(
            "Resources/Shaders/NMSUpdate.comp.txt"
        );

        ComputeShader ThresholdBorderShader(
            "Resources/Shaders/ThresholdBorder.comp.txt"
        );

        ComputeShader KeypointCompactShader(
            "Resources/Shaders/KeypointCompact.comp.txt"
        );

        ComputeShader SampleDescriptorsShader(
            "Resources/Shaders/SampleDescriptors.comp.txt"
        );


        //-----------Check Shader Programs---------//

        if (GrayShader.getProgram() == 0 ||
            ToTensorShader.getProgram() == 0 ||
            Conv1aReLUShader.getProgram() == 0 ||
            ConvArrayReLUShader.getProgram() == 0 ||
            ConvArrayShader.getProgram() == 0 ||
            MaxPoolShader.getProgram() == 0 ||
            DescriptorL2Shader.getProgram() == 0 ||
            DetectorSoftmaxShader.getProgram() == 0 ||
            NMSInitShader.getProgram() == 0 ||
            NMSDilateShader.getProgram() == 0 ||
            NMSUpdateShader.getProgram() == 0 ||
            ThresholdBorderShader.getProgram() == 0 ||
            KeypointCompactShader.getProgram() == 0 ||
            SampleDescriptorsShader.getProgram() == 0) {

            std::cout << "ERROR: Failed to initialize one or more compute shaders!" << std::endl;

            computeReady = false;
        }


        //-----------Get Uniform Locations---------//

        GLint inputChannelsReLULocation = -1;
        GLint outputChannelsReLULocation = -1;
        GLint kernelSizeReLULocation = -1;

        GLint inputChannelsLocation = -1;
        GLint outputChannelsLocation = -1;
        GLint kernelSizeLocation = -1;

        GLint epsilonLocation = -1;

        GLint nmsInitRadiusLocation = -1;
        GLint nmsDilateRadiusLocation = -1;
        GLint nmsUpdateRadiusLocation = -1;

        GLint thresholdLocation = -1;
        GLint borderLocation = -1;

        GLint keypointCapacityLocation = -1;
        GLint sampleKeypointCountLocation = -1;


        if (computeReady) {

            inputChannelsReLULocation = glGetUniformLocation(
                ConvArrayReLUShader.getProgram(),
                "uInputChannels"
            );

            outputChannelsReLULocation = glGetUniformLocation(
                ConvArrayReLUShader.getProgram(),
                "uOutputChannels"
            );

            kernelSizeReLULocation = glGetUniformLocation(
                ConvArrayReLUShader.getProgram(),
                "uKernelSize"
            );


            inputChannelsLocation = glGetUniformLocation(
                ConvArrayShader.getProgram(),
                "uInputChannels"
            );

            outputChannelsLocation = glGetUniformLocation(
                ConvArrayShader.getProgram(),
                "uOutputChannels"
            );

            kernelSizeLocation = glGetUniformLocation(
                ConvArrayShader.getProgram(),
                "uKernelSize"
            );


            epsilonLocation = glGetUniformLocation(
                DescriptorL2Shader.getProgram(),
                "uEpsilon"
            );


            nmsInitRadiusLocation = glGetUniformLocation(
                NMSInitShader.getProgram(),
                "uNmsRadius"
            );

            nmsDilateRadiusLocation = glGetUniformLocation(
                NMSDilateShader.getProgram(),
                "uNmsRadius"
            );

            nmsUpdateRadiusLocation = glGetUniformLocation(
                NMSUpdateShader.getProgram(),
                "uNmsRadius"
            );


            thresholdLocation = glGetUniformLocation(
                ThresholdBorderShader.getProgram(),
                "uThreshold"
            );

            borderLocation = glGetUniformLocation(
                ThresholdBorderShader.getProgram(),
                "uBorder"
            );


            keypointCapacityLocation = glGetUniformLocation(
                KeypointCompactShader.getProgram(),
                "uCapacity"
            );

            sampleKeypointCountLocation = glGetUniformLocation(
                SampleDescriptorsShader.getProgram(),
                "uKeypointCount"
            );


            if (inputChannelsReLULocation == -1 ||
                outputChannelsReLULocation == -1 ||
                kernelSizeReLULocation == -1 ||
                inputChannelsLocation == -1 ||
                outputChannelsLocation == -1 ||
                kernelSizeLocation == -1 ||
                epsilonLocation == -1 ||
                nmsInitRadiusLocation == -1 ||
                nmsDilateRadiusLocation == -1 ||
                nmsUpdateRadiusLocation == -1 ||
                thresholdLocation == -1 ||
                borderLocation == -1 ||
                keypointCapacityLocation == -1 ||
                sampleKeypointCountLocation == -1) {

                std::cout << "ERROR: One or more compute shader uniforms were not found!" << std::endl;

                computeReady = false;
            }
        }


        // =====================================================================
        // Load CNN Weights
        // =====================================================================

        const char* layerNames[12] = {
            "conv1a",
            "conv1b",
            "conv2a",
            "conv2b",
            "conv3a",
            "conv3b",
            "conv4a",
            "conv4b",
            "convDa",
            "convDb",
            "convPa",
            "convPb"
        };

        const int layerCin[12] = {
            1, 64, 64, 64, 64, 128, 128, 128, 128, 256, 128, 256
        };

        const int layerCout[12] = {
            64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 65
        };

        const int layerK[12] = {
            3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1
        };


        GLuint weightsBuffer[12] = {};
        GLuint biasesBuffer[12] = {};

        bool buffersCreated = false;


        if (computeReady) {

            std::ifstream weightsFile(
                "Resources/Weights/weights.bin",
                std::ios::binary
            );

            if (!weightsFile.is_open()) {

                std::cout << "FAILED TO OPEN WEIGHTS FILE" << std::endl;

                computeReady = false;
            }
            else {

                glGenBuffers(12, weightsBuffer);
                glGenBuffers(12, biasesBuffer);

                buffersCreated = true;

                long long totalParams = 0;

                std::cout.precision(8);

                for (int l = 0; l < 12; l++) {

                    const size_t weightCount =
                        static_cast<size_t>(layerCout[l]) *
                        layerCin[l] *
                        layerK[l] *
                        layerK[l];

                    const size_t biasCount =
                        static_cast<size_t>(layerCout[l]);

                    std::vector<float> w(weightCount);
                    std::vector<float> b(biasCount);

                    weightsFile.read(
                        reinterpret_cast<char*>(w.data()),
                        weightCount * sizeof(float)
                    );

                    weightsFile.read(
                        reinterpret_cast<char*>(b.data()),
                        biasCount * sizeof(float)
                    );

                    if (!weightsFile) {

                        std::cout << "weights.bin is too short at layer "
                            << layerNames[l] << std::endl;

                        computeReady = false;

                        break;
                    }


                    //-----------Upload Weights---------//

                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER,
                        weightsBuffer[l]
                    );

                    glBufferData(
                        GL_SHADER_STORAGE_BUFFER,
                        weightCount * sizeof(float),
                        w.data(),
                        GL_STATIC_DRAW
                    );


                    //-----------Upload Biases---------//

                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER,
                        biasesBuffer[l]
                    );

                    glBufferData(
                        GL_SHADER_STORAGE_BUFFER,
                        biasCount * sizeof(float),
                        b.data(),
                        GL_STATIC_DRAW
                    );


                    totalParams += static_cast<long long>(
                        weightCount + biasCount
                        );

                    std::cout << layerNames[l]
                        << " : weights ("
                        << layerCout[l] << ", "
                        << layerCin[l] << ", "
                        << layerK[l] << ", "
                        << layerK[l] << ")"
                        << "  first weight : " << w[0]
                        << "  first bias : " << b[0]
                        << std::endl;
                }


                if (computeReady) {

                    if (weightsFile.peek() != EOF) {
                        std::cout << "WARNING : weights.bin has extra data after the last layer" << std::endl;
                    }

                    std::cout << "Total parameters : " << totalParams << std::endl;
                }

                weightsFile.close();
            }
        }


        // =====================================================================
        // Keypoint and Descriptor SSBOs
        // =====================================================================

        GLuint keypointCounterBuffer = 0;
        GLuint keypointCandidateBuffer = 0;
        GLuint sampledDescriptorBuffer = 0;

        bool keypointBuffersCreated = false;


        if (computeReady) {

            glGenBuffers(1, &keypointCounterBuffer);
            glGenBuffers(1, &keypointCandidateBuffer);
            glGenBuffers(1, &sampledDescriptorBuffer);

            if (keypointCounterBuffer == 0 ||
                keypointCandidateBuffer == 0 ||
                sampledDescriptorBuffer == 0) {

                std::cout << "ERROR: Failed to generate keypoint SSBOs!" << std::endl;

                computeReady = false;
            }
            else {

                //-----------Keypoint Counter---------//

                GLuint zero = 0;

                glBindBuffer(
                    GL_SHADER_STORAGE_BUFFER,
                    keypointCounterBuffer
                );

                glBufferData(
                    GL_SHADER_STORAGE_BUFFER,
                    sizeof(GLuint),
                    &zero,
                    GL_DYNAMIC_DRAW
                );


                //-----------Candidate Keypoints---------//

                glBindBuffer(
                    GL_SHADER_STORAGE_BUFFER,
                    keypointCandidateBuffer
                );

                glBufferData(
                    GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(candidateBufferSize),
                    NULL,
                    GL_DYNAMIC_COPY
                );


                //-----------Sampled Descriptors---------//

                glBindBuffer(
                    GL_SHADER_STORAGE_BUFFER,
                    sampledDescriptorBuffer
                );

                glBufferData(
                    GL_SHADER_STORAGE_BUFFER,
                    static_cast<GLsizeiptr>(descriptorBufferSize),
                    NULL,
                    GL_DYNAMIC_COPY
                );


                keypointBuffersCreated = true;
            }
        }


        // =====================================================================
        // CNN Execution
        // =====================================================================

        if (computeReady) {

            for (int i = 0; i < imageCounter; i++) {

                //-----------Load Image---------//

                if (i != 0) {

                    imageData = stbi_load(
                        imagePaths[i].c_str(),
                        &imageWidth,
                        &imageHeight,
                        &imageChannels,
                        4
                    );
                }

                if (!imageData) {

                    std::cout << "Failed to load image number "
                        << i << std::endl;

                    continue;
                }

                std::cout << "\nImage number " << i
                    << " Loaded : '" << imagePaths[i]
                    << "' Width : " << imageWidth
                    << " Height : " << imageHeight
                    << " Channels : " << imageChannels
                    << std::endl;


                //-----------Check Image Dimensions---------//

                if (imageWidth != textureWidth ||
                    imageHeight != textureHeight) {

                    std::cout << "Skipping image: dimensions do not match texture dimensions." << std::endl;

                    stbi_image_free(imageData);
                    imageData = nullptr;

                    continue;
                }


                //-----------Reset Ping-Pong Textures---------//

                pingATexture.resize(
                    textureHeight,
                    textureWidth,
                    64
                );

                pingBTexture.resize(
                    textureHeight,
                    textureWidth,
                    64
                );


                //-----------Upload Input Image---------//

                glBindTexture(
                    GL_TEXTURE_2D,
                    inputTexture.getTextureID()
                );

                glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    imageWidth,
                    imageHeight,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    imageData
                );

                glBindTexture(GL_TEXTURE_2D, 0);


                // =====================================================================
                // Image Preprocessing
                // =====================================================================


                //-----------RGB2Gray---------//

                GrayShader.use();

                glBindImageTexture(
                    0,
                    inputTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_RGBA8
                );

                glBindImageTexture(
                    1,
                    grayTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_RGBA8
                );

                glDispatchCompute(
                    (imageWidth + 15) / 16,
                    (imageHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------ToTensor---------//

                ToTensorShader.use();

                glBindImageTexture(
                    0,
                    grayTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_RGBA8
                );

                glBindImageTexture(
                    1,
                    tensorTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (imageWidth + 15) / 16,
                    (imageHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Convolutional Backbone - Block 1
                // =====================================================================


                //-----------Conv1a + ReLU---------//

                Conv1aReLUShader.use();

                glBindImageTexture(
                    0,
                    tensorTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(
                    GL_SHADER_STORAGE_BUFFER,
                    0,
                    weightsBuffer[0]
                );

                glBindBufferBase(
                    GL_SHADER_STORAGE_BUFFER,
                    1,
                    biasesBuffer[0]
                );

                glDispatchCompute(
                    (imageWidth + 15) / 16,
                    (imageHeight + 15) / 16,
                    64
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------Conv1b + ReLU---------//

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 64);
                glUniform1i(outputChannelsReLULocation, 64);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[1]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[1]);

                glDispatchCompute(
                    (imageWidth + 15) / 16,
                    (imageHeight + 15) / 16,
                    64
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------MaxPool 1---------//

                pingATexture.resize(height1, width1, 64);

                MaxPoolShader.use();

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (width1 + 15) / 16,
                    (height1 + 15) / 16,
                    64
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Convolutional Backbone - Block 2
                // =====================================================================


                //-----------Conv2a + ReLU---------//

                pingBTexture.resize(height1, width1, 64);

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 64);
                glUniform1i(outputChannelsReLULocation, 64);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[2]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[2]);

                glDispatchCompute(
                    (width1 + 15) / 16,
                    (height1 + 15) / 16,
                    64
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------Conv2b + ReLU---------//

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 64);
                glUniform1i(outputChannelsReLULocation, 64);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[3]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[3]);

                glDispatchCompute(
                    (width1 + 15) / 16,
                    (height1 + 15) / 16,
                    64
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------MaxPool 2---------//

                pingBTexture.resize(height2, width2, 64);

                MaxPoolShader.use();

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (width2 + 15) / 16,
                    (height2 + 15) / 16,
                    64
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Convolutional Backbone - Block 3
                // =====================================================================


                //-----------Conv3a + ReLU---------//

                pingATexture.resize(height2, width2, 128);

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 64);
                glUniform1i(outputChannelsReLULocation, 128);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[4]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[4]);

                glDispatchCompute(
                    (width2 + 15) / 16,
                    (height2 + 15) / 16,
                    128
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------Conv3b + ReLU---------//

                pingBTexture.resize(height2, width2, 128);

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 128);
                glUniform1i(outputChannelsReLULocation, 128);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[5]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[5]);

                glDispatchCompute(
                    (width2 + 15) / 16,
                    (height2 + 15) / 16,
                    128
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------MaxPool 3---------//

                pingATexture.resize(height3, width3, 128);

                MaxPoolShader.use();

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    128
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Convolutional Backbone - Block 4
                // =====================================================================


                //-----------Conv4a + ReLU---------//

                pingBTexture.resize(height3, width3, 128);

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 128);
                glUniform1i(outputChannelsReLULocation, 128);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[6]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[6]);

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    128
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------Conv4b + ReLU---------//

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 128);
                glUniform1i(outputChannelsReLULocation, 128);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[7]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[7]);

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    128
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Descriptor Head
                // =====================================================================


                //-----------ConvDa + ReLU---------//

                pingBTexture.resize(height3, width3, 256);

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 128);
                glUniform1i(outputChannelsReLULocation, 256);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[8]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[8]);

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    256
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------ConvDb (No ReLU)---------//

                ConvArrayShader.use();

                glUniform1i(inputChannelsLocation, 256);
                glUniform1i(outputChannelsLocation, 256);
                glUniform1i(kernelSizeLocation, 1);

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    descriptorTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[9]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[9]);

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    256
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Detector Head
                // =====================================================================


                //-----------ConvPa + ReLU---------//

                ConvArrayReLUShader.use();

                glUniform1i(inputChannelsReLULocation, 128);
                glUniform1i(outputChannelsReLULocation, 256);
                glUniform1i(kernelSizeReLULocation, 3);

                glBindImageTexture(
                    0,
                    pingATexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[10]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[10]);

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    256
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------ConvPb (No ReLU)---------//

                ConvArrayShader.use();

                glUniform1i(inputChannelsLocation, 256);
                glUniform1i(outputChannelsLocation, 65);
                glUniform1i(kernelSizeLocation, 1);

                glBindImageTexture(
                    0,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    detectorTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer[11]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer[11]);

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    65
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Descriptor L2 Normalization
                // =====================================================================


                //-----------Descriptor L2 Normalization---------//

                DescriptorL2Shader.use();

                glUniform1f(
                    epsilonLocation,
                    1e-12f
                );

                glBindImageTexture(
                    0,
                    descriptorTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    pingBTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Detector Softmax + Heatmap Reconstruction
                // =====================================================================


                //-----------Detector Softmax + Heatmap---------//

                DetectorSoftmaxShader.use();

                glBindImageTexture(
                    0,
                    detectorTexture.getTextureID(),
                    0,
                    GL_TRUE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    heatmapTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (width3 + 15) / 16,
                    (height3 + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Non-Maximum Suppression
                // =====================================================================


                //-----------NMS Initialization---------//

                NMSInitShader.use();

                glUniform1i(
                    nmsInitRadiusLocation,
                    nmsRadius
                );

                glBindImageTexture(
                    0,
                    heatmapTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    nmsMaskATexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------NMS Iteration 1 : Dilate---------//

                NMSDilateShader.use();

                glUniform1i(
                    nmsDilateRadiusLocation,
                    nmsRadius
                );

                glBindImageTexture(
                    0,
                    nmsMaskATexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    nmsSuppressionTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------NMS Iteration 1 : Update---------//

                NMSUpdateShader.use();

                glUniform1i(
                    nmsUpdateRadiusLocation,
                    nmsRadius
                );

                glBindImageTexture(
                    0,
                    heatmapTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    nmsMaskATexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    2,
                    nmsSuppressionTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    3,
                    nmsMaskBTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------NMS Iteration 2 : Dilate---------//

                NMSDilateShader.use();

                glUniform1i(
                    nmsDilateRadiusLocation,
                    nmsRadius
                );

                glBindImageTexture(
                    0,
                    nmsMaskBTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    nmsSuppressionTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                //-----------NMS Iteration 2 : Update---------//

                NMSUpdateShader.use();

                glUniform1i(
                    nmsUpdateRadiusLocation,
                    nmsRadius
                );

                glBindImageTexture(
                    0,
                    heatmapTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    nmsMaskBTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    2,
                    nmsSuppressionTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    3,
                    nmsMaskATexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Confidence Threshold + Border Removal
                // =====================================================================


                //-----------Threshold + Border---------//

                ThresholdBorderShader.use();

                glUniform1f(
                    thresholdLocation,
                    detectionThreshold
                );

                glUniform1i(
                    borderLocation,
                    borderSize
                );

                glBindImageTexture(
                    0,
                    heatmapTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    1,
                    nmsMaskATexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindImageTexture(
                    2,
                    filteredScoreTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_WRITE_ONLY,
                    GL_R32F
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


                // =====================================================================
                // Keypoint Compaction
                // =====================================================================


                //-----------Reset Keypoint Counter---------//

                GLuint zero = 0;

                glBindBuffer(
                    GL_SHADER_STORAGE_BUFFER,
                    keypointCounterBuffer
                );

                glBufferSubData(
                    GL_SHADER_STORAGE_BUFFER,
                    0,
                    sizeof(GLuint),
                    &zero
                );


                //-----------Compact Keypoints---------//

                KeypointCompactShader.use();

                glUniform1ui(
                    keypointCapacityLocation,
                    static_cast<GLuint>(pixelCount)
                );

                glBindImageTexture(
                    0,
                    filteredScoreTexture.getTextureID(),
                    0,
                    GL_FALSE,
                    0,
                    GL_READ_ONLY,
                    GL_R32F
                );

                glBindBufferBase(
                    GL_SHADER_STORAGE_BUFFER,
                    2,
                    keypointCounterBuffer
                );

                glBindBufferBase(
                    GL_SHADER_STORAGE_BUFFER,
                    3,
                    keypointCandidateBuffer
                );

                glDispatchCompute(
                    (textureWidth + 15) / 16,
                    (textureHeight + 15) / 16,
                    1
                );

                glMemoryBarrier(
                    GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT
                );


                // =====================================================================
                // Read Compact Keypoints
                // =====================================================================


                //-----------Read Keypoint Count---------//

                GLuint detectedKeypointCount = 0;

                glBindBuffer(
                    GL_SHADER_STORAGE_BUFFER,
                    keypointCounterBuffer
                );

                glGetBufferSubData(
                    GL_SHADER_STORAGE_BUFFER,
                    0,
                    sizeof(GLuint),
                    &detectedKeypointCount
                );


                if (detectedKeypointCount > pixelCount) { detectedKeypointCount = static_cast<GLuint>(pixelCount);
                }


                std::cout << "Detected keypoints before top-K: "
                    << detectedKeypointCount << std::endl;


                //-----------Read Candidate Keypoints---------//

                std::vector<GLuint> candidateData(
                    static_cast<size_t>(detectedKeypointCount) * 2
                );

                if (detectedKeypointCount > 0) {

                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER,
                        keypointCandidateBuffer
                    );

                    glGetBufferSubData(
                        GL_SHADER_STORAGE_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(
                            candidateData.size() * sizeof(GLuint)
                            ),
                        candidateData.data()
                    );
                }


                //-----------Decode and Rank Keypoints---------//

                std::vector<std::tuple<float, GLuint, GLuint>> keypointCandidates;

                keypointCandidates.reserve(detectedKeypointCount);

                for (GLuint k = 0; k < detectedKeypointCount; k++) {

                    GLuint linearIndex = candidateData[k * 2];

                    float score = 0.0f;

                    std::memcpy(
                        &score,
                        &candidateData[k * 2 + 1],
                        sizeof(float)
                    );

                    GLuint x = linearIndex % static_cast<GLuint>(textureWidth);
                    GLuint y = linearIndex / static_cast<GLuint>(textureWidth);

                    keypointCandidates.emplace_back(
                        score,
                        x,
                        y
                    );
                }


                //-----------Sort by Confidence---------//

                std::sort(
                    keypointCandidates.begin(),
                    keypointCandidates.end(),
                    std::greater<std::tuple<float, GLuint, GLuint>>()
                );


                //-----------Apply Maximum Keypoint Count---------//

                if (keypointCandidates.size() > maxKeypoints) {

                    keypointCandidates.resize(maxKeypoints);
                }

                const GLuint selectedKeypointCount =
                    static_cast<GLuint>(keypointCandidates.size());


                std::cout << "Selected keypoints after top-K: "
                    << selectedKeypointCount << std::endl;


                // =====================================================================
                // Upload Selected Keypoints
                // =====================================================================


                std::vector<GLuint> keypointCoordinates(
                    static_cast<size_t>(selectedKeypointCount) * 2
                );

                for (GLuint k = 0; k < selectedKeypointCount; k++) {

                    keypointCoordinates[k * 2] =
                        std::get<1>(keypointCandidates[k]);

                    keypointCoordinates[k * 2 + 1] =
                        std::get<2>(keypointCandidates[k]);
                }


                if (selectedKeypointCount > 0) {

                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER,
                        keypointCandidateBuffer
                    );

                    glBufferSubData(
                        GL_SHADER_STORAGE_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(
                            keypointCoordinates.size() * sizeof(GLuint)
                            ),
                        keypointCoordinates.data()
                    );
                }


                // =====================================================================
                // Bilinear Descriptor Sampling
                // =====================================================================


                std::vector<float> sampledDescriptors(
                    static_cast<size_t>(selectedKeypointCount) *
                    descriptorDimension
                );


                if (selectedKeypointCount > 0) {

                    //-----------Sample Descriptors---------//

                    SampleDescriptorsShader.use();

                    glUniform1ui(
                        sampleKeypointCountLocation,
                        selectedKeypointCount
                    );

                    glBindImageTexture(
                        0,
                        pingBTexture.getTextureID(),
                        0,
                        GL_TRUE,
                        0,
                        GL_READ_ONLY,
                        GL_R32F
                    );

                    glBindBufferBase(
                        GL_SHADER_STORAGE_BUFFER,
                        2,
                        keypointCandidateBuffer
                    );

                    glBindBufferBase(
                        GL_SHADER_STORAGE_BUFFER,
                        3,
                        sampledDescriptorBuffer
                    );

                    glDispatchCompute(
                        (selectedKeypointCount + 63) / 64,
                        1,
                        1
                    );

                    glMemoryBarrier(
                        GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_BUFFER_UPDATE_BARRIER_BIT
                    );


                    //-----------Read Sampled Descriptors---------//

                    glBindBuffer(
                        GL_SHADER_STORAGE_BUFFER,
                        sampledDescriptorBuffer
                    );

                    glGetBufferSubData(
                        GL_SHADER_STORAGE_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(
                            sampledDescriptors.size() * sizeof(float)
                            ),
                        sampledDescriptors.data()
                    );
                }


                // =====================================================================
                // Save SuperPoint Features
                // =====================================================================


                //-----------Feature File---------//

                std::filesystem::path inputPath(imagePaths[i]);

                std::string outputFileName =
                    inputPath.stem().string() + "_superpoint.bin";

                std::filesystem::path outputPath =
                    std::filesystem::path("Resources/Features") / outputFileName;


                std::ofstream featureFile(
                    outputPath,
                    std::ios::binary
                );


                if (!featureFile.is_open()) {

                    std::cout << "ERROR: Could not create feature file: "
                        << outputPath.string() << std::endl;
                }
                else {

                    //-----------Feature File Header---------//

                    const uint32_t magic = 0x53505054;
                    const uint32_t version = 1;

                    const uint32_t numberOfKeypoints =
                        selectedKeypointCount;

                    const uint32_t descriptorSize =
                        descriptorDimension;


                    featureFile.write(
                        reinterpret_cast<const char*>(&magic),
                        sizeof(uint32_t)
                    );

                    featureFile.write(
                        reinterpret_cast<const char*>(&version),
                        sizeof(uint32_t)
                    );

                    featureFile.write(
                        reinterpret_cast<const char*>(&numberOfKeypoints),
                        sizeof(uint32_t)
                    );

                    featureFile.write(
                        reinterpret_cast<const char*>(&descriptorSize),
                        sizeof(uint32_t)
                    );


                    //-----------Write Keypoints---------//

                    for (GLuint k = 0; k < selectedKeypointCount; k++) {

                        float x = static_cast<float>(
                            std::get<1>(keypointCandidates[k])
                            );

                        float y = static_cast<float>(
                            std::get<2>(keypointCandidates[k])
                            );

                        float score = std::get<0>(keypointCandidates[k]);


                        featureFile.write(
                            reinterpret_cast<const char*>(&x),
                            sizeof(float)
                        );

                        featureFile.write(
                            reinterpret_cast<const char*>(&y),
                            sizeof(float)
                        );

                        featureFile.write(
                            reinterpret_cast<const char*>(&score),
                            sizeof(float)
                        );
                    }


                    //-----------Write Descriptors---------//

                    if (!sampledDescriptors.empty()) {

                        featureFile.write(
                            reinterpret_cast<const char*>(sampledDescriptors.data()),
                            static_cast<std::streamsize>(
                                sampledDescriptors.size() * sizeof(float)
                                )
                        );
                    }


                    featureFile.close();


                    if (featureFile) {

                        std::cout << "Feature file saved: "
                            << outputPath.string() << std::endl;
                    }
                    else {

                        std::cout << "ERROR: Failed while writing feature file!"
                            << std::endl;
                    }
                }


                // =====================================================================
                // Image Results
                // =====================================================================

                std::cout << "\nSuperPoint completed for image "
                    << imagePaths[i] << std::endl;

                std::cout << "Image dimensions: "
                    << textureWidth << " x " << textureHeight << std::endl;

                std::cout << "Detector heatmap: "
                    << textureWidth << " x " << textureHeight << std::endl;

                std::cout << "NMS radius: "
                    << nmsRadius << std::endl;

                std::cout << "Detection threshold: "
                    << detectionThreshold << std::endl;

                std::cout << "Border removal: "
                    << borderSize << " pixels" << std::endl;

                std::cout << "Final keypoints: "
                    << selectedKeypointCount << std::endl;

                std::cout << "Descriptor dimension: 256" << std::endl;


                //-----------Free CPU Image---------//

                stbi_image_free(imageData);
                imageData = nullptr;
            }
        }


        //-----------Cleanup Weight Buffers---------//

        if (buffersCreated) {

            glDeleteBuffers(12, weightsBuffer);
            glDeleteBuffers(12, biasesBuffer);
        }


        //-----------Cleanup Keypoint Buffers---------//

        if (keypointBuffersCreated) {

            glDeleteBuffers(1, &keypointCounterBuffer);
            glDeleteBuffers(1, &keypointCandidateBuffer);
            glDeleteBuffers(1, &sampledDescriptorBuffer);
        }

    } // Texture and ComputeShader destructors execute here.


    //-----------Free Remaining CPU Image---------//

    if (imageData != nullptr) {

        stbi_image_free(imageData);
        imageData = nullptr;
    }


    //-----------Check VRAM After CNN---------//

    CheckGPUMemory();


    // =====================================================================
    // Dear ImGui Initialization
    // =====================================================================

    IMGUI_CHECKVERSION();

    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");


    // =====================================================================
    // Render Loop
    // =====================================================================

    while (!glfwWindowShouldClose(window)) {

        // ImGui frame start

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        // per-frame time logic

        float currentFrame = static_cast<float>(glfwGetTime());

        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;


        // input

        processInput(window);


        // render

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        glClear(
            GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT |
            GL_STENCIL_BUFFER_BIT
        );


        // ImGui rendering

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    // =====================================================================
    // Cleanup
    // =====================================================================

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}


// =====================================================================
// Input
// =====================================================================

void processInput(GLFWwindow* window)
{
    static bool escWasPressed = false;

    bool escPressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;

    if (escPressed && !escWasPressed) {

        if (cursorMode == CursorMode::Captured) {

            cursorMode = CursorMode::Free;

            glfwSetInputMode(
                window,
                GLFW_CURSOR,
                GLFW_CURSOR_NORMAL
            );
        }
        else {

            glfwSetWindowShouldClose(window, true);
        }
    }

    escWasPressed = escPressed;

    if (cursorMode != CursorMode::Captured) {
        return;
    }


    /*
    glm::vec3 currentPos = ourCamera->getPosition();
    glm::vec3 front = ourCamera->getFront();
    glm::vec3 right = glm::normalize(glm::cross(front, ourCamera->getUp()));

    float cameraSpeed = 2.5f * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        ourCamera->setPosition(currentPos.x + front.x * cameraSpeed,
            currentPos.y + front.y * cameraSpeed,
            currentPos.z + front.z * cameraSpeed);
    */
}


// =====================================================================
// GLFW Callbacks
// =====================================================================

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (cursorMode != CursorMode::Captured) {
        return;
    }

    // Camera::mouse_callback(window, xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // Camera::scroll_callback(window, xoffset, yoffset);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    ImGuiIO& io = ImGui::GetIO();

    if (io.WantCaptureMouse) {
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT &&
        action == GLFW_PRESS &&
        cursorMode == CursorMode::Free) {

        cursorMode = CursorMode::Captured;

        glfwSetInputMode(
            window,
            GLFW_CURSOR,
            GLFW_CURSOR_DISABLED
        );
    }
}


// =====================================================================
// Texture Loading Utility
// =====================================================================

unsigned int loadTexture(char const* path)
{
    unsigned int textureID;

    glGenTextures(1, &textureID);

    int width, height, nrComponents;

    unsigned char* data = stbi_load(
        path,
        &width,
        &height,
        &nrComponents,
        0
    );

    if (data) {

        GLenum format = GL_RGB;

        if (nrComponents == 1) {
            format = GL_RED;
        }
        else if (nrComponents == 3) {
            format = GL_RGB;
        }
        else if (nrComponents == 4) {
            format = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            format,
            width,
            height,
            0,
            format,
            GL_UNSIGNED_BYTE,
            data
        );

        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MIN_FILTER,
            GL_LINEAR_MIPMAP_LINEAR
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_LINEAR
        );

        stbi_image_free(data);
    }
    else {

        std::cout << "Texture failed to load at path: "
            << path << std::endl;

        stbi_image_free(data);
    }

    return textureID;
}
