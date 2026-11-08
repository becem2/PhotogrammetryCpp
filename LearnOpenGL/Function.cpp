// Windows / DXGI (must come before glad.h, which Function.h pulls in)
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifdef _WIN32
#include <Windows.h>
#include <dxgi1_4.h>
#endif

#include "Function.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(_MSC_VER)
#pragma comment(lib, "dxgi.lib")
#endif


namespace
{
    constexpr int kPreviewGap = 8;          // pixels between the two images of a match picture


    // Nearest multiple of 8, at least 8
    int roundToMultipleOf8(float value)
    {
        const int rounded = static_cast<int>(std::lround(value / 8.0f)) * 8;
        return std::max(8, rounded);
    }


    //-----------Tiny RGB Canvas Drawing---------//

    void canvasPixel(std::vector<unsigned char>& rgb, int cw, int ch,
        int x, int y, const unsigned char* color)
    {
        if (x < 0 || y < 0 || x >= cw || y >= ch) {
            return;
        }

        unsigned char* p = rgb.data() + (static_cast<size_t>(y) * cw + x) * 3;

        p[0] = color[0];
        p[1] = color[1];
        p[2] = color[2];
    }


    void canvasDisc(std::vector<unsigned char>& rgb, int cw, int ch,
        int cx, int cy, int radius, const unsigned char* color)
    {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx * dx + dy * dy <= radius * radius) {
                    canvasPixel(rgb, cw, ch, cx + dx, cy + dy, color);
                }
            }
        }
    }


    void canvasLine(std::vector<unsigned char>& rgb, int cw, int ch,
        float x0, float y0, float x1, float y1, const unsigned char* color)
    {
        const float dx = x1 - x0;
        const float dy = y1 - y0;

        const int steps = std::max(1, static_cast<int>(std::max(std::fabs(dx), std::fabs(dy))));

        for (int i = 0; i <= steps; i++) {

            const float t = static_cast<float>(i) / static_cast<float>(steps);

            canvasDisc(rgb, cw, ch,
                static_cast<int>(std::lround(x0 + dx * t)),
                static_cast<int>(std::lround(y0 + dy * t)),
                1, color);
        }
    }
}


//-----------System---------//

void Function::checkGPUMemory()
{
#ifdef _WIN32

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

#else

    std::cout << "\nGPU memory report is only available on Windows (DXGI)." << std::endl;

#endif
}


std::string Function::formatDuration(double seconds)
{
    if (seconds < 0.0) {
        seconds = 0.0;
    }

    // Work in whole milliseconds so rounding can never print "60.000 s"
    long long milliseconds = static_cast<long long>(std::llround(seconds * 1000.0));

    const long long hours = milliseconds / 3600000;
    milliseconds %= 3600000;

    const long long minutes = milliseconds / 60000;
    milliseconds %= 60000;

    std::ostringstream text;

    if (hours > 0) {
        text << hours << " h ";
    }

    if (hours > 0 || minutes > 0) {
        text << minutes << " min ";
    }

    text << std::fixed << std::setprecision(3) << milliseconds / 1000.0 << " s";

    return text.str();
}


//-----------File System---------//

bool Function::collectImagePaths(const std::string& folderPath,
    std::vector<std::string>& imagePaths,
    int& totalFileCount)
{
    imagePaths.clear();
    totalFileCount = 0;

    std::filesystem::directory_entry folderEntry(folderPath);

    if (!folderEntry.exists() || !folderEntry.is_directory()) {
        std::cout << "ERROR : " << folderPath
            << " is not a directory or it doesn't exist" << std::endl;

        return false;
    }

    for (const auto& imageEntry : std::filesystem::directory_iterator(folderPath)) {

        if (!imageEntry.is_regular_file()) {
            continue;
        }

        std::string extension = imageEntry.path().extension().string();

        for (char& c : extension) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        if (extension == ".png" ||
            extension == ".jpg" ||
            extension == ".jpeg") {

            imagePaths.push_back(imageEntry.path().string());
        }

        totalFileCount++;
    }

    std::sort(imagePaths.begin(), imagePaths.end());

    return true;
}


bool Function::createDirectory(const std::string& path)
{
    std::error_code error;

    std::filesystem::create_directories(path, error);

    if (error) {
        std::cout << "ERROR: Could not create " << path
            << " directory: " << error.message() << std::endl;

        return false;
    }

    return true;
}


//-----------Image Sizes And Resizing---------//

ImageSize Function::computeNetworkSize(ImageSize source, int maxLongSide)
{
    const int longSide = std::max(source.width, source.height);

    const float scale = longSide > maxLongSide
        ? static_cast<float>(maxLongSide) / static_cast<float>(longSide)
        : 1.0f;

    ImageSize network;

    network.width = roundToMultipleOf8(source.width * scale);
    network.height = roundToMultipleOf8(source.height * scale);

    return network;
}


ImageSize Function::scaleSize(ImageSize size, float scale)
{
    ImageSize scaled;

    scaled.width = std::max(1, static_cast<int>(std::lround(size.width * scale)));
    scaled.height = std::max(1, static_cast<int>(std::lround(size.height * scale)));

    return scaled;
}


void Function::resizeAreaRGBA(const unsigned char* src, ImageSize srcSize,
    unsigned char* dst, ImageSize dstSize)
{
    const int srcW = srcSize.width;
    const int srcH = srcSize.height;
    const int dstW = dstSize.width;
    const int dstH = dstSize.height;

    const float scaleX = static_cast<float>(srcW) / static_cast<float>(dstW);
    const float scaleY = static_cast<float>(srcH) / static_cast<float>(dstH);

    for (int y = 0; y < dstH; y++) {

        const float y0 = y * scaleY;
        const float y1 = y0 + scaleY;
        const int iy0 = static_cast<int>(std::floor(y0));
        const int iy1 = std::min(srcH, static_cast<int>(std::ceil(y1)));

        for (int x = 0; x < dstW; x++) {

            const float x0 = x * scaleX;
            const float x1 = x0 + scaleX;
            const int ix0 = static_cast<int>(std::floor(x0));
            const int ix1 = std::min(srcW, static_cast<int>(std::ceil(x1)));

            float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float weightSum = 0.0f;

            for (int iy = iy0; iy < iy1; iy++) {

                const float wy = std::min(y1, static_cast<float>(iy + 1)) - std::max(y0, static_cast<float>(iy));

                for (int ix = ix0; ix < ix1; ix++) {

                    const float wx = std::min(x1, static_cast<float>(ix + 1)) - std::max(x0, static_cast<float>(ix));
                    const float w = wx * wy;

                    const unsigned char* p = src + (static_cast<size_t>(iy) * srcW + ix) * 4;

                    sum[0] += p[0] * w;
                    sum[1] += p[1] * w;
                    sum[2] += p[2] * w;
                    sum[3] += p[3] * w;

                    weightSum += w;
                }
            }

            unsigned char* q = dst + (static_cast<size_t>(y) * dstW + x) * 4;

            for (int c = 0; c < 4; c++) {
                q[c] = static_cast<unsigned char>(sum[c] / weightSum + 0.5f);
            }
        }
    }
}


//-----------OpenGL---------//

unsigned int Function::loadTexture(const char* path)
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


//-----------Match Visualization---------//

bool Function::saveMatchVisualization(const std::string& path,
    const unsigned char* rgba0, const unsigned char* rgba1,
    ImageSize previewSize, float scaleX, float scaleY,
    const SuperPointFeatures& features0, const SuperPointFeatures& features1,
    const SuperGlueMatches& matches)
{
    const int width = previewSize.width;
    const int height = previewSize.height;

    const int canvasWidth = width * 2 + kPreviewGap;
    const int canvasHeight = height;
    const int offsetX1 = width + kPreviewGap;

    std::vector<unsigned char> canvas(static_cast<size_t>(canvasWidth) * canvasHeight * 3, 20);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            const size_t src = (static_cast<size_t>(y) * width + x) * 4;

            unsigned char* left = canvas.data() + (static_cast<size_t>(y) * canvasWidth + x) * 3;
            unsigned char* right = canvas.data() + (static_cast<size_t>(y) * canvasWidth + offsetX1 + x) * 3;

            left[0] = rgba0[src]; left[1] = rgba0[src + 1]; left[2] = rgba0[src + 2];
            right[0] = rgba1[src]; right[1] = rgba1[src + 1]; right[2] = rgba1[src + 2];
        }
    }

    const unsigned char unmatchedColor[3] = { 230, 60, 60 };
    const unsigned char whiteColor[3] = { 255, 255, 255 };

    // Keypoints that were used by SuperGlue but found no partner: small red dots
    for (int i = 0; i < matches.keypointCount0 && i < static_cast<int>(features0.keypoints.size()); i++) {
        if (i < static_cast<int>(matches.matches0.size()) && matches.matches0[i] < 0) {
            canvasDisc(canvas, canvasWidth, canvasHeight,
                static_cast<int>(std::lround(features0.keypoints[i].x * scaleX)),
                static_cast<int>(std::lround(features0.keypoints[i].y * scaleY)),
                1, unmatchedColor);
        }
    }

    for (int i = 0; i < matches.keypointCount1 && i < static_cast<int>(features1.keypoints.size()); i++) {
        if (i < static_cast<int>(matches.matches1.size()) && matches.matches1[i] < 0) {
            canvasDisc(canvas, canvasWidth, canvasHeight,
                offsetX1 + static_cast<int>(std::lround(features1.keypoints[i].x * scaleX)),
                static_cast<int>(std::lround(features1.keypoints[i].y * scaleY)),
                1, unmatchedColor);
        }
    }

    // Matches: line colour goes from red (low confidence) to green (high confidence).
    // Drawn from the weakest to the strongest so the best matches end up on top.
    std::vector<Match> sorted = matches.pairs;

    std::sort(sorted.begin(), sorted.end(),
        [](const Match& a, const Match& b) { return a.score < b.score; });

    for (const Match& m : sorted) {

        if (m.index0 < 0 || m.index0 >= static_cast<int>(features0.keypoints.size()) ||
            m.index1 < 0 || m.index1 >= static_cast<int>(features1.keypoints.size())) {
            continue;
        }

        const float t = std::min(1.0f, std::max(0.0f, (m.score - 0.2f) / 0.8f));

        const unsigned char color[3] = {
            static_cast<unsigned char>(255.0f * (1.0f - t)),
            static_cast<unsigned char>(60.0f + 195.0f * t),
            50
        };

        const float x0 = features0.keypoints[m.index0].x * scaleX;
        const float y0 = features0.keypoints[m.index0].y * scaleY;
        const float x1 = offsetX1 + features1.keypoints[m.index1].x * scaleX;
        const float y1 = features1.keypoints[m.index1].y * scaleY;

        canvasLine(canvas, canvasWidth, canvasHeight, x0, y0, x1, y1, color);

        canvasDisc(canvas, canvasWidth, canvasHeight,
            static_cast<int>(std::lround(x0)), static_cast<int>(std::lround(y0)), 3, whiteColor);
        canvasDisc(canvas, canvasWidth, canvasHeight,
            static_cast<int>(std::lround(x1)), static_cast<int>(std::lround(y1)), 3, whiteColor);

        canvasDisc(canvas, canvasWidth, canvasHeight,
            static_cast<int>(std::lround(x0)), static_cast<int>(std::lround(y0)), 2, color);
        canvasDisc(canvas, canvasWidth, canvasHeight,
            static_cast<int>(std::lround(x1)), static_cast<int>(std::lround(y1)), 2, color);
    }

    return stbi_write_jpg(path.c_str(), canvasWidth, canvasHeight, 3, canvas.data(), 92) != 0;
}