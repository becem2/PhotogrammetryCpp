#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "SuperPoint.h"


//-----------Configuration---------//

struct TiledConfig
{
    // Size of one tile (and therefore of the SuperPoint object). Multiples of 8.
    // 1600 x 1200 is a size the network already handles comfortably (< 1 GB of VRAM).
    int tileWidth = 1600;
    int tileHeight = 1200;

    // Pixels around the part of a tile that the tile "owns". Keypoints closer than this to an
    // INNER tile edge are discarded (the tile next door sees them with full context).
    // The network sees about 84 px around a point, so 48 leaves only negligible edge effects.
    // Multiple of 8, and 2 * margin must be smaller than the tile.
    int margin = 48;

    // Keypoints kept after merging (best scores first). 0 = keep all.
    size_t maxKeypoints = 8192;

    // Keypoints closer than this (square window) to a better one are removed after merging,
    // which also fixes the seams between tiles. Use the SuperPoint nmsRadius. 0 = off.
    int nmsRadius = 4;

    bool verbose = true;
};


//-----------Tile Geometry---------//

struct TilePlan
{
    int originX = 0, originY = 0;                // top-left pixel of the tile in the image
    int coreX0 = 0, coreX1 = 0;                  // columns the tile owns: [coreX0, coreX1)
    int coreY0 = 0, coreY1 = 0;                  // rows the tile owns:    [coreY0, coreY1)
};


namespace TiledExtractor
{
    // Size of the SuperPoint object to create for an image of this size:
    // the configured tile, or the whole image when it is smaller.
    // Image sides must be multiples of 8.
    void tileSizeFor(int imageWidth, int imageHeight, const TiledConfig& config, int& tileWidth, int& tileHeight);

    // Tiles covering the image. The "cores" partition the image exactly (every pixel is owned
    // by one tile) and every core keeps at least `margin` pixels of context to each inner tile edge.
    std::vector<TilePlan> planTiles(int imageWidth, int imageHeight, int tileWidth, int tileHeight, int margin);


    // Extraction of one tile: rgba is tileWidth * tileHeight * 4 bytes.
    using ExtractFunction = std::function<bool(const unsigned char* rgba, SuperPointFeatures& features)>;

    // Generic version (used by the tests, or with any extractor).
    bool extract(const ExtractFunction& extractTile, int tileWidth, int tileHeight,
        const unsigned char* rgba, int imageWidth, int imageHeight,
        const TiledConfig& config, SuperPointFeatures& features);

    // SuperPoint version. `network` must have been initialized with tileSizeFor(...).
    // rgba is the WHOLE image (imageWidth * imageHeight * 4 bytes). The result has the same
    // format as SuperPoint::extract (global pixel coordinates, sorted by descending score,
    // L2-normalized descriptors), so SuperGlue and the .bin writer work unchanged.
    bool extract(SuperPoint& network, const unsigned char* rgba, int imageWidth, int imageHeight,
        const TiledConfig& config, SuperPointFeatures& features);
}
