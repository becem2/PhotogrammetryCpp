#define NOMINMAX
#define GLFW_INCLUDE_NONE

// GLAD / GLFW / IMGUI
#include <glad.h>
#include <glfw3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// C++ STL
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// stb_image
#include "stb_image.h"

// My Classes
#include "Function.h"
#include "SuperPoint.h"
#include "SuperGlue.h"


// =====================================================================
// Settings
// =====================================================================

namespace Settings
{
    // Folders
    constexpr const char* imageFolder = "Resources/Images";
    constexpr const char* featureFolder = "Resources/Features";
    constexpr const char* matchFolder = "Resources/Matches";
    constexpr const char* visualizationFolder = "Resources/Visualizations";

    // Photos are downscaled so that their longer side is at most this many pixels.
    // SuperPoint keeps 64-channel full-resolution feature maps in VRAM, so running it
    // on 6000 x 4000 needs ~12 GB for the ping-pong textures alone. Around 1600 px
    // it needs well under 1 GB.
    constexpr int maxLongSide = 1600;

    // Match pictures are drawn at this fraction of the network resolution
    constexpr float previewScale = 0.5f;

    // false : match each image with the next one (0-1, 1-2, 2-3, ...)
    // true  : match every image with every later image
    constexpr bool matchAllPairs = false;
}


// =====================================================================
// Window / Input State
// =====================================================================

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void processInput(GLFWwindow* window);


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
// Image Set (everything the pipeline stages share)
// =====================================================================

struct ImageSet
{
    std::vector<std::string> paths;     // every image found, sorted

    ImageSize source;                   // full photo size (all photos must match the first one)
    ImageSize network;                  // SuperPoint input size (downscaled, multiple of 8)
    ImageSize preview;                  // match picture thumbnail size

    // Filled by extractFeatures, same order in all three
    std::vector<std::string> names;
    std::vector<SuperPointFeatures> features;
    std::vector<std::vector<unsigned char>> previews;    // RGBA thumbnails
};


// =====================================================================
// Pipeline Stages
// =====================================================================

//-----------Find The Images And Decide The Sizes---------//

static bool loadImageSet(ImageSet& images)
{
    int filesCounter = 0;

    if (!Function::collectImagePaths(Settings::imageFolder, images.paths, filesCounter)) {
        return false;
    }

    const int imageCounter = static_cast<int>(images.paths.size());

    std::cout << "Found " << filesCounter
        << " files in the '" << Settings::imageFolder
        << "' folder " << imageCounter
        << " of them are images" << std::endl;

    for (int i = 0; i < imageCounter; i++) {
        std::cout << "Image number " << i
            << " : '" << images.paths[i] << "'" << std::endl;
    }

    if (imageCounter == 0) {
        std::cout << "No images found!" << std::endl;
        return false;
    }


    // The SuperPoint object is sized for one resolution, so the first image
    // decides the size every other image must match.

    int channels = 0;

    if (!stbi_info(images.paths[0].c_str(), &images.source.width, &images.source.height, &channels)) {
        std::cout << "Failed to read the first image header!" << std::endl;
        return false;
    }

    if (images.source.width <= 0 || images.source.height <= 0) {
        std::cout << "ERROR: Image dimensions must be positive." << std::endl;
        return false;
    }

    images.network = Function::computeNetworkSize(images.source, Settings::maxLongSide);
    images.preview = Function::scaleSize(images.network, Settings::previewScale);

    std::cout << "\nSource images : " << images.source.width << " x " << images.source.height
        << "\nNetwork input : " << images.network.width << " x " << images.network.height
        << "  (keypoint scale to full size: x "
        << static_cast<float>(images.source.width) / static_cast<float>(images.network.width)
        << ", y "
        << static_cast<float>(images.source.height) / static_cast<float>(images.network.height)
        << ")" << std::endl;

    return true;
}


//-----------SuperPoint: Keypoints + Descriptors Of Every Image---------//

static void extractFeatures(ImageSet& images)
{
    //-----------Configuration---------//

    SuperPointConfig config;

    config.nmsRadius = 4;
    config.borderSize = 4;
    config.detectionThreshold = 0.005f;
    config.maxKeypoints = 4096;

    config.shaderDirectory = "Resources/Shaders/";
    config.weightsPath = "Resources/Weights/weights.bin";

    config.verbose = true;


    //-----------Initialization---------//

    // Creates the textures, compiles the shaders, loads weights.bin
    // and allocates the keypoint buffers.

    SuperPoint superPoint(config);

    if (!superPoint.initialize(images.network.width, images.network.height)) {
        std::cout << "ERROR: SuperPoint failed to initialize!" << std::endl;
        return;
    }

    std::vector<unsigned char> networkPixels(
        static_cast<size_t>(images.network.width) * images.network.height * 4);


    //-----------Process Every Image---------//

    for (size_t i = 0; i < images.paths.size(); i++) {

        const std::string& path = images.paths[i];

        // Load (always as RGBA, which is what SuperPoint::extract expects)

        int imageWidth = 0;
        int imageHeight = 0;
        int imageChannels = 0;

        unsigned char* imageData = stbi_load(path.c_str(), &imageWidth, &imageHeight, &imageChannels, 4);

        if (!imageData) {
            std::cout << "Failed to load image number " << i << std::endl;
            continue;
        }

        std::cout << "\nImage number " << i
            << " Loaded : '" << path
            << "' Width : " << imageWidth
            << " Height : " << imageHeight
            << " Channels : " << imageChannels
            << std::endl;

        if (imageWidth != images.source.width || imageHeight != images.source.height) {
            std::cout << "Skipping image: dimensions do not match the first image." << std::endl;
            stbi_image_free(imageData);
            continue;
        }


        // Downscale to the network resolution

        if (images.network.width == imageWidth && images.network.height == imageHeight) {
            std::memcpy(networkPixels.data(), imageData, networkPixels.size());
        }
        else {
            Function::resizeAreaRGBA(imageData, images.source, networkPixels.data(), images.network);
        }

        stbi_image_free(imageData);


        // Run SuperPoint

        SuperPointFeatures features;

        if (!superPoint.extract(networkPixels.data(), features)) {
            std::cout << "ERROR: SuperPoint failed on image number " << i << std::endl;
            continue;
        }


        // Save the features

        const std::filesystem::path inputPath(path);

        const std::filesystem::path outputPath =
            std::filesystem::path(Settings::featureFolder) /
            (inputPath.stem().string() + "_superpoint.bin");

        if (SuperPoint::save(outputPath.string(), features)) {
            std::cout << "Feature file saved: " << outputPath.string() << std::endl;
        }
        else {
            std::cout << "ERROR: Failed while writing feature file!" << std::endl;
        }


        // Results

        const SuperPointConfig& used = superPoint.getConfig();

        std::cout << "\nSuperPoint completed for image " << path << std::endl;

        std::cout << "Image dimensions (network input): "
            << features.imageWidth << " x " << features.imageHeight
            << "  (full photo " << images.source.width << " x " << images.source.height << ")" << std::endl;

        std::cout << "NMS radius: " << used.nmsRadius << std::endl;
        std::cout << "Detection threshold: " << used.detectionThreshold << std::endl;
        std::cout << "Border removal: " << used.borderSize << " pixels" << std::endl;
        std::cout << "Final keypoints: " << features.count() << std::endl;
        std::cout << "Descriptor dimension: " << SuperPointFeatures::DescriptorDimension << std::endl;


        // Keep everything SuperGlue and the match pictures need.
        // networkPixels still holds this image at network resolution.

        images.previews.emplace_back(
            static_cast<size_t>(images.preview.width) * images.preview.height * 4);

        Function::resizeAreaRGBA(networkPixels.data(), images.network,
            images.previews.back().data(), images.preview);

        images.names.push_back(inputPath.stem().string());
        images.features.push_back(std::move(features));
    }

} // SuperPoint is destroyed here, while the OpenGL context is still current.


//-----------SuperGlue: Match The Images And Draw The Result---------//

static void matchImages(const ImageSet& images)
{
    if (images.features.size() < 2) {
        std::cout << "\nSuperGlue skipped: at least two images with features are required." << std::endl;
        return;
    }


    //-----------Configuration---------//

    SuperGlueConfig config;

    config.sinkhornIterations = 100;
    config.matchThreshold = 0.2f;
    config.maxKeypoints = 1024;

    config.shaderDirectory = "Resources/Shaders/";
    config.weightsPath = "Resources/Weights/superglue_indoor.bin";   // or superglue_outdoor.bin

    config.verbose = true;


    //-----------Initialization---------//

    SuperGlue superGlue(config);

    if (!Function::createDirectory(Settings::matchFolder)) {
        return;
    }

    if (!superGlue.initialize()) {
        std::cout << "ERROR: SuperGlue failed to initialize!" << std::endl;
        return;
    }

    // Only a warning if this fails: the match files are still written
    Function::createDirectory(Settings::visualizationFolder);

    // Network-resolution keypoint coordinates -> preview coordinates
    const float previewScaleX = static_cast<float>(images.preview.width) / static_cast<float>(images.network.width);
    const float previewScaleY = static_cast<float>(images.preview.height) / static_cast<float>(images.network.height);


    //-----------Match Image Pairs---------//

    for (size_t a = 0; a + 1 < images.features.size(); a++) {

        const size_t lastB = Settings::matchAllPairs ? images.features.size() - 1 : a + 1;

        for (size_t b = a + 1; b <= lastB; b++) {

            const std::string pairName = images.names[a] + "_" + images.names[b];

            std::cout << "\nSuperGlue: '" << images.names[a]
                << "' <-> '" << images.names[b] << "'" << std::endl;

            SuperGlueMatches matches;

            if (!superGlue.match(images.features[a], images.features[b], matches)) {
                std::cout << "ERROR: SuperGlue failed on this pair." << std::endl;
                continue;
            }


            // Match file (.bin)

            const std::filesystem::path matchPath =
                std::filesystem::path(Settings::matchFolder) / (pairName + "_superglue.bin");

            if (SuperGlue::save(matchPath.string(), matches)) {
                std::cout << "Match file saved: " << matchPath.string() << std::endl;
            }
            else {
                std::cout << "ERROR: Failed while writing match file!" << std::endl;
            }

            std::cout << "Keypoints used: "
                << matches.keypointCount0 << " / "
                << matches.keypointCount1 << std::endl;

            std::cout << "Matches: " << matches.count() << std::endl;


            // Match picture (.jpg)

            const std::filesystem::path picturePath =
                std::filesystem::path(Settings::visualizationFolder) / (pairName + "_matches.jpg");

            if (Function::saveMatchVisualization(picturePath.string(),
                images.previews[a].data(), images.previews[b].data(),
                images.preview, previewScaleX, previewScaleY,
                images.features[a], images.features[b], matches)) {

                std::cout << "Match picture saved: " << picturePath.string() << std::endl;
            }
            else {
                std::cout << "ERROR: Failed while writing match picture!" << std::endl;
            }
        }
    }

} // SuperGlue is destroyed here, while the OpenGL context is still current.


//-----------Timing Summary---------//

static void printTimingSummary(double superPointSeconds, double superGlueSeconds)
{
    std::cout << "\n=============== Execution Time ===============" << std::endl;

    std::cout << "SuperPoint : " << Function::formatDuration(superPointSeconds) << std::endl;
    std::cout << "SuperGlue  : " << Function::formatDuration(superGlueSeconds) << std::endl;
    std::cout << "Combined   : " << Function::formatDuration(superPointSeconds + superGlueSeconds) << std::endl;

    std::cout << "==============================================" << std::endl;
}


// =====================================================================
// Window
// =====================================================================

static GLFWwindow* createWindow()
{
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    screenWidth = 800;
    screenHeight = 600;

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "LearnOpenGL", NULL, NULL);

    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    // configure global OpenGL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    return window;
}


//-----------Dear ImGui Render Loop (also destroys the window)---------//

static void runRenderLoop(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();

    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}


// =====================================================================
// MAIN
// =====================================================================

int main()
{
    GLFWwindow* window = createWindow();

    if (window == nullptr) {
        return -1;
    }

    Function::checkGPUMemory();


    //-----------Prepare---------//

    ImageSet images;

    if (!loadImageSet(images) ||
        !Function::createDirectory(Settings::featureFolder)) {

        glfwDestroyWindow(window);
        glfwTerminate();

        return -1;
    }


    //-----------Pipeline: SuperPoint -> SuperGlue---------//

    // Each stage is timed on its own. SuperPoint's time includes loading, resizing
    // and saving every image; SuperGlue's includes loading the weights, matching
    // and drawing the match pictures.

    using Clock = std::chrono::steady_clock;

    const auto superPointStart = Clock::now();

    extractFeatures(images);

    const auto superPointEnd = Clock::now();

    matchImages(images);

    const auto superGlueEnd = Clock::now();

    const double superPointSeconds = std::chrono::duration<double>(superPointEnd - superPointStart).count();
    const double superGlueSeconds = std::chrono::duration<double>(superGlueEnd - superPointEnd).count();

    Function::checkGPUMemory();


    //-----------Window---------//

    runRenderLoop(window);


    //-----------End Of Program---------//

    printTimingSummary(superPointSeconds, superGlueSeconds);

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