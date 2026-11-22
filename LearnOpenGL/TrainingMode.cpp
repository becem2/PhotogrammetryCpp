#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "TrainingMode.h"
#include "Settings.h"
#include "Function.h"
#include "SuperPoint.h"
#include "TrainDetector.h"
#include "TrainingData.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace TrainingMode
{
	// ---------Folder Checks---------
	bool validateTrainingData(const TrainingConfig& config)
	{
		std::cout << "\n=============== Validating Training Data ===============" << std::endl;

		const std::filesystem::path labelsPath(config.pseudoLabelsFolder);

		if (!std::filesystem::exists(labelsPath)) {
			std::cout << "ERROR: Pseudo-labels folder not found: " << config.pseudoLabelsFolder << std::endl;
			std::cout << "Hint: Run mode 2 (Pseudo-Labels) first to generate training labels." << std::endl;
			return false;
		}

		const std::vector<std::filesystem::path> samples = TrainingData::listSamples(config.pseudoLabelsFolder);

		std::cout << "Found " << samples.size() << " .sample files in: " << config.pseudoLabelsFolder << std::endl;

		if (samples.empty()) {
			std::cout << "ERROR: No pseudo-label files found!" << std::endl;
			std::cout << "Hint: Run mode 2 (Pseudo-Labels) first to generate training labels." << std::endl;
			return false;
		}

		// The original photos are not needed any more: every .sample contains its own crop.
		if (std::filesystem::exists(config.trainingImagesFolder)) {
			std::cout << "Training images folder present (not read by training): "
				<< config.trainingImagesFolder << std::endl;
		}

		std::cout << "======================================================" << std::endl;
		std::cout << "Training data validation: OK" << std::endl;

		return true;
	}


	void printTrainingInfo(const TrainingConfig& config)
	{
		std::cout << "\n=============== Training Configuration ===============" << std::endl;
		std::cout << "Pseudo-Labels Folder   : " << config.pseudoLabelsFolder << std::endl;
		std::cout << "Starting Weights       : " << config.inputWeightsPath << std::endl;
		std::cout << "Output Weights Path    : " << config.outputWeightsPath << std::endl;
		std::cout << "Network Resolution     : " << config.networkWidth << " x " << config.networkHeight << std::endl;
		std::cout << "Learning Rate          : " << config.learningRate << std::endl;
		std::cout << "Epochs                 : " << config.epochCount << std::endl;
		std::cout << "Batch Size             : " << config.batchSize << std::endl;
		std::cout << "Early stopping patience: " << config.patience << (config.patience == 0 ? " (off)" : "") << std::endl;
		std::cout << "Validation fraction    : " << config.validationFraction << " (of the images)" << std::endl;
		std::cout << "======================================================" << std::endl;
	}


	// ---------Training---------
	bool runTraining(const TrainingConfig& config)
	{
		std::cout << "\n=============== SuperPoint Detector Training Mode ===============" << std::endl;

		if (!validateTrainingData(config)) {
			return false;
		}

		TrainingConfig used = config;

		// The network must have exactly the size of the crops: take it from the first .sample
		const std::vector<std::filesystem::path> samples = TrainingData::listSamples(used.pseudoLabelsFolder);

		int sampleWidth = 0, sampleHeight = 0;

		if (TrainingData::readSize(samples.front(), sampleWidth, sampleHeight) &&
			(sampleWidth != used.networkWidth || sampleHeight != used.networkHeight)) {

			std::cout << "Note: the crops are " << sampleWidth << " x " << sampleHeight
				<< ", not " << used.networkWidth << " x " << used.networkHeight
				<< ". Using the crop size." << std::endl;

			used.networkWidth = sampleWidth;
			used.networkHeight = sampleHeight;
		}

		printTrainingInfo(used);

		// The output folder must exist before the first save
		const std::filesystem::path outPath(used.outputWeightsPath);

		if (outPath.has_parent_path() && !Function::createDirectory(outPath.parent_path().string())) {
			return false;
		}

		// ---------SuperPoint with the PRETRAINED weights---------
		SuperPointConfig spConfig;
		spConfig.weightsPath = used.inputWeightsPath;
		spConfig.shaderDirectory = "Resources/Shaders/";
		spConfig.verbose = used.verbose;

		SuperPoint superPoint(spConfig);

		if (!superPoint.initialize(used.networkWidth, used.networkHeight)) {
			std::cout << "ERROR: SuperPoint failed to initialize!" << std::endl;
			return false;
		}

		// ---------Train the detector head---------
		TrainDetectorOptions options;
		options.dataFolder = used.pseudoLabelsFolder;
		options.outputWeights = used.outputWeightsPath;
		options.epochs = used.epochCount;
		options.batchSize = used.batchSize;
		options.learningRate = used.learningRate;
		options.patience = used.patience;
		options.validationFraction = used.validationFraction;
		options.seed = used.seed;

		std::cout << "\n>>> Training <<<" << std::endl;

		const bool ok = trainDetector(superPoint, options);

		if (ok) {
			std::cout << "\nTraining finished. Best weights: " << used.outputWeightsPath << std::endl;
			std::cout << "Use them for inference by setting SuperPointConfig::weightsPath." << std::endl;
		}

		return ok;

	} // SuperPoint is destroyed here, while the OpenGL context is still current.
}
