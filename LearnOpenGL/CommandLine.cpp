#include "CommandLine.h"
#include "SuperPoint.h"
#include "Settings.h"

#include <iostream>
#include <stdexcept>

namespace CommandLineParser
{
	bool parseCommandLine(int argc, char** argv, CommandLine& cli)
	{
		cli.options.outputFolder = "Resources/PseudoLabels";

		try {
			for (int i = 1; i < argc; i++) {

				const std::string arg = argv[i];

				auto next = [&]() -> const char* {
					if (i + 1 >= argc) {
						throw std::runtime_error("missing value after " + arg);
					}
					return argv[++i];
				};

				if (arg == "--mode")                { 
					std::string modeStr = next();
					if (modeStr == "feature") cli.mode = ApplicationMode::FeatureDetection;
					else if (modeStr == "pseudo") cli.mode = ApplicationMode::PseudoLabels;
					else if (modeStr == "training") cli.mode = ApplicationMode::Training;
					else throw std::runtime_error("unknown mode: " + modeStr);
				}
				else if (arg == "--pseudo-labels")        { cli.pseudoLabels = true; cli.mode = ApplicationMode::PseudoLabels; cli.options.inputFolder = next(); }
				else if (arg == "--out")             { cli.options.outputFolder = next(); }
				else if (arg == "--weights")         { cli.weights = next(); }
				else if (arg == "--crop-w")          { cli.cropWidth = std::stoi(next()); }
				else if (arg == "--crop-h")          { cli.cropHeight = std::stoi(next()); }
				else if (arg == "--crops")           { cli.options.cropsPerImage = std::stoi(next()); }
				else if (arg == "--homographies")    { cli.options.homographies = std::stoi(next()); }
				else if (arg == "--det-thresh")      { cli.options.detectionThreshold = std::stof(next()); }
				else if (arg == "--min-valid")       { cli.options.minValid = std::stof(next()); }
				else if (arg == "--start")           { cli.options.startIndex = std::stoi(next()); }
				else if (arg == "--end")             { cli.options.endIndex = std::stoi(next()); }
				else if (arg == "--seed")            { cli.options.seed = static_cast<unsigned int>(std::stoul(next())); }
				else if (arg == "--resume")          { cli.options.resume = true; }
				else if (arg == "--no-vis")          { cli.options.saveVisualization = false; }
				else if (!arg.empty() && arg[0] != '-' &&
				         cli.mode == ApplicationMode::PseudoLabels && cli.options.inputFolder.empty()) {
					cli.options.inputFolder = arg;      // --mode pseudo <imageFolder>
				}
				else {
					throw std::runtime_error("unknown argument " + arg);
				}
			}
		}
		catch (const std::exception& e) {
			std::cout << "ERROR: bad command line: " << e.what() << std::endl;
			return false;
		}

		if (cli.mode == ApplicationMode::PseudoLabels) cli.pseudoLabels = true;

		if (cli.pseudoLabels) {

			if (cli.cropWidth <= 0 || cli.cropHeight <= 0 || cli.cropWidth % 8 != 0 || cli.cropHeight % 8 != 0) {
				std::cout << "ERROR: --crop-w and --crop-h must be positive multiples of 8." << std::endl;
				return false;
			}

			if (cli.options.cropsPerImage <= 0 || cli.options.homographies <= 0) {
				std::cout << "ERROR: --crops and --homographies must be positive." << std::endl;
				return false;
			}
		}

		return true;
	}


	bool runPseudoLabels(const CommandLine& cli)
	{
		PseudoLabelOptions options = cli.options;

		if (options.inputFolder.empty()) {
			options.inputFolder = Settings::trainingImagesFolder;     // Resources/TrainingImages
			std::cout << "No image folder given, using " << options.inputFolder << std::endl;
		}

		SuperPointConfig config;

		config.nmsRadius = options.nmsRadius;
		config.borderSize = options.borderSize;
		config.shaderDirectory = "Resources/Shaders/";
		config.weightsPath = cli.weights;     // must be the PRETRAINED weights
		config.verbose = false;

		SuperPoint superPoint(config);

		if (!superPoint.initialize(cli.cropWidth, cli.cropHeight)) {
			std::cout << "ERROR: SuperPoint failed to initialize!" << std::endl;
			return false;
		}

		return makePseudoLabels(superPoint, options);

	} // SuperPoint is destroyed here, while the OpenGL context is still current.


	ApplicationMode selectApplicationMode()
	{
		std::cout << "\n=============== Select Application Mode ===============" << std::endl;
		std::cout << "1. Feature Detection & Matching (extract features and match image pairs)" << std::endl;
		std::cout << "2. Pseudo-Labels Generation (generate training labels with homographic adaptation)" << std::endl;
		std::cout << "3. Training Mode (train SuperPoint detector head)" << std::endl;
		std::cout << "======================================================" << std::endl;
		std::cout << "Enter mode (1-3) [default: 1]: ";

		std::string input;
		std::getline(std::cin, input);

		if (input.empty()) {
			return ApplicationMode::FeatureDetection;
		}

		try {
			int choice = std::stoi(input);
			switch (choice) {
				case 1: return ApplicationMode::FeatureDetection;
				case 2: return ApplicationMode::PseudoLabels;
				case 3: return ApplicationMode::Training;
				default:
					std::cout << "Invalid choice. Defaulting to Feature Detection." << std::endl;
					return ApplicationMode::FeatureDetection;
			}
		}
		catch (const std::exception& e) {
			std::cout << "Invalid input. Defaulting to Feature Detection." << std::endl;
			return ApplicationMode::FeatureDetection;
		}
	}
}
