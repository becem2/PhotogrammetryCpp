#pragma once

#include <vector>
#include <string>
#include "SuperPoint.h"
#include "Function.h"

// Everything the pipeline stages need to share
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

namespace Pipeline
{
	// Pipeline stages
	bool loadImageSet(ImageSet& images);
	void extractFeatures(ImageSet& images);
	void matchImages(const ImageSet& images);
	void printTimingSummary(double superPointSeconds, double superGlueSeconds);

	// Mode execution functions
	bool runFeatureDetectionMode();
	bool runTrainingMode();
}
