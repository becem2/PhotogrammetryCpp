#include "Pipeline.h"
#include "Settings.h"
#include "Function.h"
#include "SuperGlue.h"
#include "stb_image.h"
#include "TrainingMode.h"
#include "TiledExtractor.h"

#include <iostream>
#include <filesystem>
#include <cstring>
#include <chrono>

namespace Pipeline
{
	// ---------Find The Images And Decide The Sizes---------
	bool loadImageSet(ImageSet& images)
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


	// ---------SuperPoint: Keypoints + Descriptors Of Every Image---------
	void extractFeatures(ImageSet& images)
	{
		// ---------Configuration---------
		SuperPointConfig config;

		config.nmsRadius = 4;
		config.borderSize = 4;
		config.detectionThreshold = 0.005f;
		config.maxKeypoints = 4096;

		config.shaderDirectory = "Resources/Shaders/";

		// The fine-tuned detector is used as soon as Mode 3 has produced it
		config.weightsPath = std::filesystem::exists(Settings::trainedWeights)
			? Settings::trainedWeights : Settings::superPointWeights;

		std::cout << "SuperPoint weights: " << config.weightsPath << std::endl;

		config.verbose = false;     // one summary line per photo is printed by the tiled extractor


		// ---------Initialization---------
		// Creates the textures, compiles the shaders, loads weights.bin
		// and allocates the keypoint buffers.

		// The network is created for one TILE (or for the whole photo when it is smaller)
		// and slid over the photo: VRAM and the length of every GPU command stay small at 4K.
		TiledConfig tiling;

		tiling.tileWidth = Settings::tileWidth;
		tiling.tileHeight = Settings::tileHeight;
		tiling.margin = Settings::tileMargin;
		tiling.maxKeypoints = static_cast<size_t>(Settings::maxMergedKeypoints);
		tiling.nmsRadius = config.nmsRadius;
		tiling.verbose = true;

		int tileWidth = 0;
		int tileHeight = 0;

		TiledExtractor::tileSizeFor(images.network.width, images.network.height, tiling, tileWidth, tileHeight);

		std::cout << "SuperPoint tile : " << tileWidth << " x " << tileHeight
			<< "  (photo " << images.network.width << " x " << images.network.height << ")" << std::endl;

		SuperPoint superPoint(config);

		if (!superPoint.initialize(tileWidth, tileHeight)) {
			std::cout << "ERROR: SuperPoint failed to initialize!" << std::endl;
			return;
		}

		std::vector<unsigned char> networkPixels(
			static_cast<size_t>(images.network.width) * images.network.height * 4);


		// ---------Process Every Image---------
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

			if (!TiledExtractor::extract(superPoint, networkPixels.data(),
				images.network.width, images.network.height, tiling, features)) {
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


	// ---------SuperGlue: Match The Images And Draw The Result---------
	void matchImages(const ImageSet& images)
	{
		if (images.features.size() < 2) {
			std::cout << "\nSuperGlue skipped: at least two images with features are required." << std::endl;
			return;
		}


		// ---------Configuration---------
		SuperGlueConfig config;

		config.sinkhornIterations = 100;
		config.matchThreshold = 0.2f;
		config.maxKeypoints = 1024;

		config.shaderDirectory = "Resources/Shaders/";
		config.weightsPath = "Resources/Weights/superglue_indoor.bin";   // or superglue_outdoor.bin

		config.verbose = true;


		// ---------Initialization---------
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


		// ---------Match Image Pairs---------
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


	// ---------Timing Summary---------
	void printTimingSummary(double superPointSeconds, double superGlueSeconds)
	{
		std::cout << "\n=============== Execution Time ===============" << std::endl;

		std::cout << "SuperPoint : " << Function::formatDuration(superPointSeconds) << std::endl;
		std::cout << "SuperGlue  : " << Function::formatDuration(superGlueSeconds) << std::endl;
		std::cout << "Combined   : " << Function::formatDuration(superPointSeconds + superGlueSeconds) << std::endl;

		std::cout << "==============================================" << std::endl;
	}


	// ---------Feature Detection Mode---------
	bool runFeatureDetectionMode()
	{
		std::cout << "\n=============== Feature Detection Mode ===============" << std::endl;

		// Prepare
		ImageSet images;

		if (!loadImageSet(images) ||
			!Function::createDirectory(Settings::featureFolder)) {
			return false;
		}

		// Run pipeline with timing
		using Clock = std::chrono::steady_clock;

		const auto superPointStart = Clock::now();

		extractFeatures(images);

		const auto superPointEnd = Clock::now();

		matchImages(images);

		const auto superGlueEnd = Clock::now();

		const double superPointSeconds = std::chrono::duration<double>(superPointEnd - superPointStart).count();
		const double superGlueSeconds = std::chrono::duration<double>(superGlueEnd - superPointEnd).count();

		Function::checkGPUMemory();

		// Print summary
		printTimingSummary(superPointSeconds, superGlueSeconds);

		return true;
	}


	// ---------Training Mode---------
	bool runTrainingMode()
	{
		std::cout << "\n=============== Training Mode ===============" << std::endl;

		// Create training configuration
		TrainingMode::TrainingConfig trainConfig;
		trainConfig.trainingImagesFolder = Settings::trainingImagesFolder;
		trainConfig.pseudoLabelsFolder = Settings::pseudoLabelsFolder;
		trainConfig.networkWidth = Settings::defaultNetworkWidth;
		trainConfig.networkHeight = Settings::defaultNetworkHeight;

		// Run training
		return TrainingMode::runTraining(trainConfig);
	}
}
