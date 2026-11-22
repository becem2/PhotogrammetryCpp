#pragma once

#include "PseudoLabels.h"

// Application modes
enum class ApplicationMode
{
	FeatureDetection,    // Normal feature extraction and matching
	PseudoLabels,        // Generate pseudo-labels for training
	Training             // Train SuperPoint detector head
};

struct CommandLine
{
	ApplicationMode mode = ApplicationMode::FeatureDetection;
	bool pseudoLabels = false;  // For backwards compatibility with old CLI
	PseudoLabelOptions options;
	int cropWidth = 1024;
	int cropHeight = 768;
	std::string weights = "Resources/Weights/weights.bin";
};

namespace CommandLineParser
{
	bool parseCommandLine(int argc, char** argv, CommandLine& cli);
	bool runPseudoLabels(const CommandLine& cli);
	ApplicationMode selectApplicationMode();
}
