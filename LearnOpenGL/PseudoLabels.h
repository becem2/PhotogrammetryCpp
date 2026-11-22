#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "SuperPoint.h"
#include "TrainingData.h"


//-----------Options---------//

struct PseudoLabelOptions
{
    std::string inputFolder;        // images (.png / .jpg / .jpeg), any colour: they are read as gray
    std::string outputFolder;       // receives <image>_c<k>.sample (+ _labels.bmp)

    int cropsPerImage = 4;
    int homographies = 50;          // views per crop, including the unwarped one (the paper uses 100)

    float detectionThreshold = 0.02f;   // on the AVERAGED heatmap
    float minValid = 0.3f;              // a pixel must be visible in this fraction of the views
    int erode = 8;                      // pixels removed around the warped image border

    int nmsRadius = 4;                  // keep these equal to what extract_keypoints() uses in Python
    int borderSize = 4;

    // Random homography
    float scaleAmplitude = 0.15f;
    float maxAngle = 1.5707963f;        // pi / 2
    float translation = 0.1f;
    float perspective = 0.08f;

    int startIndex = 0;             // first image (numbered from 0, sorted by path)
    int endIndex = -1;              // stop BEFORE this image; -1 = the last one
    bool resume = false;            // skip images whose crops already exist
    bool saveVisualization = true;  // <name>_labels.bmp : the crop with the labels drawn on it (LOOK AT THESE)

    unsigned int seed = 0;
};


//-----------Building Blocks (also usable on their own)---------//

// Returns heatmap = width * height floats for an RGBA8 image of the network size.
using HeatmapFunction = std::function<bool(const unsigned char* rgba, std::vector<float>& heatmap)>;

// Averages the detector heatmap over many random warps of `gray` (width * height bytes).
//   aggregated : the Homographic-Adaptation heatmap
//   single     : heatmap of the unwarped image alone
bool homographicAdaptation(const std::vector<unsigned char>& gray, int width, int height,
    const HeatmapFunction& detect, const PseudoLabelOptions& options, std::mt19937_64& rng,
    std::vector<float>& aggregated, std::vector<float>& single);

// Threshold + greedy non-maximum suppression (best first, square window) + border removal.
std::vector<LabelPoint> detectKeypoints(const std::vector<float>& heatmap, int width, int height,
    float threshold, int nmsRadius, int borderSize);


//-----------Whole Stage---------//

// Needs an INITIALIZED SuperPoint with the PRETRAINED weights and the crop size
// (e.g. initialize(640, 480)). Crops have exactly the network size; images smaller
// than that are skipped. Needs a current OpenGL context.
bool makePseudoLabels(SuperPoint& network, const PseudoLabelOptions& options);
