#pragma once

#include <string>

// Training mode configuration and execution
namespace TrainingMode
{
	// Training configuration
	struct TrainingConfig
	{
		// Only the .sample files are needed for training (each one holds its own gray crop
		// and labels). The images folder is just reported when it exists.
		std::string trainingImagesFolder = "Resources/TrainingImages";
		std::string pseudoLabelsFolder = "Resources/PseudoLabels";

		std::string inputWeightsPath = "Resources/Weights/weights.bin";          // pretrained start
		std::string outputWeightsPath = "Resources/Weights/weights_trained.bin"; // best epoch (+ ".csv" log)

		int networkWidth = 640;     // replaced by the size stored in the .sample files if they differ
		int networkHeight = 480;

		float learningRate = 1e-4f;
		int epochCount = 10;
		int batchSize = 4;

		int patience = 0;                   // stop after this many epochs without improvement, 0 = never
		float validationFraction = 0.1f;    // share of IMAGES kept for validation
		unsigned int seed = 0;

		bool verbose = true;
	};

	// Execute training mode: trains the detector head (convPa + convPb) on the GPU
	bool runTraining(const TrainingConfig& config);

	// Helper functions
	bool validateTrainingData(const TrainingConfig& config);
	void printTrainingInfo(const TrainingConfig& config);
}
