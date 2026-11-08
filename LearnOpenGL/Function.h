#pragma once

#include <string>
#include <vector>

#include "SuperPoint.h"
#include "SuperGlue.h"


//-----------Image Size---------//

struct ImageSize
{
    int width = 0;
    int height = 0;
};


//-----------Function (complementary helpers)---------//
//
// Everything the pipeline needs that is not SuperPoint / SuperGlue themselves:
// GPU memory report, folder and image scanning, image resizing and the
// match pictures. All methods are static, there is nothing to instantiate.

class Function
{
public:

    Function() = delete;


    //-----------System---------//

    // Prints the GPU name, dedicated VRAM and the application's memory budget / usage.
    // Windows only (DXGI); on other systems it prints a short notice instead.
    static void checkGPUMemory();

    // "12.345 s", "1 min 23.456 s" or "1 h 2 min 3.456 s"
    static std::string formatDuration(double seconds);


    //-----------File System---------//

    // Collects every .png / .jpg / .jpeg file of the folder, sorted by path.
    // totalFileCount receives the number of regular files found (images or not).
    // Returns false (after printing the reason) if the folder does not exist.
    static bool collectImagePaths(const std::string& folderPath,
        std::vector<std::string>& imagePaths,
        int& totalFileCount);

    // Creates the directory (and its parents). Prints the reason and returns false on failure.
    static bool createDirectory(const std::string& path);


    //-----------Image Sizes And Resizing---------//

    // Network input size: longer side <= maxLongSide, both sides multiples of 8
    // (SuperPoint requires it). Images that are already small enough are not enlarged.
    static ImageSize computeNetworkSize(ImageSize source, int maxLongSide);

    // size * scale, rounded, never smaller than 1 pixel
    static ImageSize scaleSize(ImageSize size, float scale);

    // Area-average (box filter) downscale of an RGBA8 image. Averaging every source pixel
    // that falls inside each destination pixel avoids the aliasing of plain bilinear
    // sampling, so the corners SuperPoint finds stay stable.
    // dst must hold dstSize.width * dstSize.height * 4 bytes.
    static void resizeAreaRGBA(const unsigned char* src, ImageSize srcSize,
        unsigned char* dst, ImageSize dstSize);


    //-----------OpenGL---------//

    // Loads an image file into a new OpenGL texture and returns its id.
    static unsigned int loadTexture(const char* path);


    //-----------Match Visualization---------//

    // Writes a JPG with the two images side by side and the SuperGlue matches drawn on it:
    // lines go from red (low confidence) to green (high confidence), unmatched keypoints
    // are small red dots.
    //
    // rgba0 / rgba1 : the two preview images (previewSize.width * previewSize.height * 4 bytes)
    // scaleX/scaleY : network-resolution keypoint coordinates -> preview coordinates
    static bool saveMatchVisualization(const std::string& path,
        const unsigned char* rgba0, const unsigned char* rgba1,
        ImageSize previewSize, float scaleX, float scaleY,
        const SuperPointFeatures& features0, const SuperPointFeatures& features1,
        const SuperGlueMatches& matches);
};