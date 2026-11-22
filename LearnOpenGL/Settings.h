#pragma once

namespace Settings
{
	// Input/Output Folders
	constexpr const char* imageFolder = "Resources/Images";
	constexpr const char* featureFolder = "Resources/Features";
	constexpr const char* matchFolder = "Resources/Matches";
	constexpr const char* visualizationFolder = "Resources/Visualizations";

	// Training Folders
	constexpr const char* pseudoLabelsFolder = "Resources/PseudoLabels";
	constexpr const char* trainingImagesFolder = "Resources/TrainingImages";

	// Photos are downscaled so that their longer side is at most this many pixels.
	// 4096 keeps 4K photos (3840 x 2160, 4000 x 3000, 4096 x ...) at full resolution.
	// SuperPoint then runs on TILES (see below), so VRAM no longer depends on the photo size.
	constexpr int maxLongSide = 4096;

	// Tiled SuperPoint: the network is created for ONE tile and slid over the photo.
	// 1600 x 1200 is a size that needs well under 1 GB of VRAM. Multiples of 8.
	constexpr int tileWidth = 1600;
	constexpr int tileHeight = 1200;

	// Keypoints closer than this to an inner tile edge are left to the neighbouring tile.
	// The network sees about 84 px around a point. Multiple of 8.
	constexpr int tileMargin = 48;

	// Keypoints kept per photo after the tiles are merged (best scores first)
	constexpr int maxMergedKeypoints = 8192;

	// SuperPoint weights: the fine-tuned file is used when it exists, otherwise the pretrained one
	constexpr const char* superPointWeights = "Resources/Weights/weights.bin";
	constexpr const char* trainedWeights = "Resources/Weights/weights_trained.bin";

	// Match pictures are drawn at this fraction of the network resolution
	constexpr float previewScale = 0.5f;

	// false : match each image with the next one (0-1, 1-2, 2-3, ...)
	// true  : match every image with every later image
	constexpr bool matchAllPairs = false;

	// Training settings: crops are taken from the photos at FULL resolution (no downscaling),
	// so the detector learns the scale it will see at inference.
	constexpr int defaultNetworkWidth = 1024;
	constexpr int defaultNetworkHeight = 768;
}
