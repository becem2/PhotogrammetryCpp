#pragma once

#include <string>

#include "SuperPoint.h"


struct TrainDetectorOptions
{
    std::string dataFolder;                                              // folder with the .sample files
    std::string outputWeights = "Resources/Weights/weights_finetuned.bin";

    int epochs = 20;
    int batchSize = 8;
    int patience = 0;                  // 0 = never stop early
    float learningRate = 1e-4f;
    float validationFraction = 0.1f;   // share of IMAGES kept for validation
    unsigned int seed = 0;
};


// Fine-tunes the detector head of an INITIALIZED SuperPoint (initialize(width, height)
// with the crop size, e.g. 640 x 480). Needs a current OpenGL context.
// Saves the best weights (lowest validation loss) in the weights.bin layout.
// NOTE: the SuperPoint object itself keeps the LAST epoch's weights; load the saved file to get the best ones.
bool trainDetector(SuperPoint& network, const TrainDetectorOptions& options);
