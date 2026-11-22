#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "TiledExtractor.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>


namespace
{
    // origin, coreBegin, coreEnd along one axis
    using AxisTile = std::array<int, 3>;

    std::vector<AxisTile> planAxis(int imageLength, int tileLength, int margin)
    {
        std::vector<AxisTile> tiles;

        if (imageLength <= tileLength) {
            tiles.push_back({ 0, 0, imageLength });
            return tiles;
        }

        const int coreLength = tileLength - 2 * margin;

        for (int begin = 0; begin < imageLength; begin += coreLength) {

            const int end = std::min(begin + coreLength, imageLength);
            const int origin = std::clamp(begin - margin, 0, imageLength - tileLength);

            tiles.push_back({ origin, begin, end });
        }

        return tiles;
    }

    struct Candidate
    {
        float score = 0.0f;
        int x = 0;
        int y = 0;
        int tile = 0;
        size_t index = 0;     // index inside that tile's keypoint list
    };
}


namespace TiledExtractor
{
    void tileSizeFor(int imageWidth, int imageHeight, const TiledConfig& config, int& tileWidth, int& tileHeight)
    {
        tileWidth = std::min(config.tileWidth, imageWidth);
        tileHeight = std::min(config.tileHeight, imageHeight);
    }


    std::vector<TilePlan> planTiles(int imageWidth, int imageHeight, int tileWidth, int tileHeight, int margin)
    {
        std::vector<TilePlan> plan;

        if (imageWidth <= 0 || imageHeight <= 0 || tileWidth <= 0 || tileHeight <= 0) return plan;

        if ((imageWidth > tileWidth && 2 * margin >= tileWidth) ||
            (imageHeight > tileHeight && 2 * margin >= tileHeight)) {
            return plan;
        }

        const std::vector<AxisTile> xs = planAxis(imageWidth, tileWidth, margin);
        const std::vector<AxisTile> ys = planAxis(imageHeight, tileHeight, margin);

        for (const AxisTile& y : ys) {
            for (const AxisTile& x : xs) {
                TilePlan t;
                t.originX = x[0]; t.coreX0 = x[1]; t.coreX1 = x[2];
                t.originY = y[0]; t.coreY0 = y[1]; t.coreY1 = y[2];
                plan.push_back(t);
            }
        }

        return plan;
    }


    bool extract(const ExtractFunction& extractTile, int tileWidth, int tileHeight,
        const unsigned char* rgba, int imageWidth, int imageHeight,
        const TiledConfig& config, SuperPointFeatures& features)
    {
        features = SuperPointFeatures();

        if (rgba == nullptr || tileWidth > imageWidth || tileHeight > imageHeight ||
            imageWidth % 8 != 0 || imageHeight % 8 != 0) {
            std::cerr << "TiledExtractor: the image sides must be multiples of 8 and at least as large as the tile ("
                << imageWidth << " x " << imageHeight << " image, " << tileWidth << " x " << tileHeight << " tile)." << std::endl;
            return false;
        }

        const std::vector<TilePlan> plan = planTiles(imageWidth, imageHeight, tileWidth, tileHeight, config.margin);

        if (plan.empty()) {
            std::cerr << "TiledExtractor: invalid tile layout (margin " << config.margin
                << " is too large for a " << tileWidth << " x " << tileHeight << " tile)." << std::endl;
            return false;
        }

        constexpr size_t D = SuperPointFeatures::DescriptorDimension;

        std::vector<SuperPointFeatures> perTile(plan.size());
        std::vector<unsigned char> tilePixels(static_cast<size_t>(tileWidth) * tileHeight * 4);

        std::vector<Candidate> candidates;
        size_t rawCount = 0;

        for (size_t t = 0; t < plan.size(); t++) {

            const TilePlan& p = plan[t];

            for (int row = 0; row < tileHeight; row++) {
                std::memcpy(&tilePixels[static_cast<size_t>(row) * tileWidth * 4],
                    rgba + (static_cast<size_t>(p.originY + row) * imageWidth + p.originX) * 4,
                    static_cast<size_t>(tileWidth) * 4);
            }

            if (!extractTile(tilePixels.data(), perTile[t])) {
                std::cerr << "TiledExtractor: extraction failed on tile " << t << std::endl;
                return false;
            }

            const SuperPointFeatures& f = perTile[t];
            rawCount += f.keypoints.size();

            for (size_t i = 0; i < f.keypoints.size(); i++) {

                const int gx = p.originX + static_cast<int>(f.keypoints[i].x + 0.5f);
                const int gy = p.originY + static_cast<int>(f.keypoints[i].y + 0.5f);

                // Only the keypoints inside the core of this tile (the neighbours own the rest)
                if (gx >= p.coreX0 && gx < p.coreX1 && gy >= p.coreY0 && gy < p.coreY1) {
                    candidates.push_back({ f.keypoints[i].score, gx, gy, static_cast<int>(t), i });
                }
            }
        }

        // Best first (ties broken by position so the result never depends on the tile order)
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
            });

        // Greedy suppression across the seams, then the cap
        std::vector<unsigned char> blocked;
        if (config.nmsRadius > 0) {
            blocked.assign(static_cast<size_t>(imageWidth) * imageHeight, 0);
        }

        features.imageWidth = imageWidth;
        features.imageHeight = imageHeight;

        for (const Candidate& c : candidates) {

            if (config.maxKeypoints > 0 && features.keypoints.size() >= config.maxKeypoints) break;

            if (config.nmsRadius > 0) {

                if (blocked[static_cast<size_t>(c.y) * imageWidth + c.x]) continue;

                for (int dy = -config.nmsRadius; dy <= config.nmsRadius; dy++) {
                    const int y = c.y + dy;
                    if (y < 0 || y >= imageHeight) continue;

                    for (int dx = -config.nmsRadius; dx <= config.nmsRadius; dx++) {
                        const int x = c.x + dx;
                        if (x >= 0 && x < imageWidth) blocked[static_cast<size_t>(y) * imageWidth + x] = 1;
                    }
                }
            }

            Keypoint kp;
            kp.x = static_cast<float>(c.x);
            kp.y = static_cast<float>(c.y);
            kp.score = c.score;
            features.keypoints.push_back(kp);

            const float* d = &perTile[c.tile].descriptors[c.index * D];
            features.descriptors.insert(features.descriptors.end(), d, d + D);
        }

        if (config.verbose) {
            std::cout << "Tiled SuperPoint: " << plan.size() << " tiles of " << tileWidth << " x " << tileHeight
                << ", " << rawCount << " raw keypoints -> " << features.keypoints.size() << " after merging" << std::endl;
        }

        return true;
    }


    bool extract(SuperPoint& network, const unsigned char* rgba, int imageWidth, int imageHeight,
        const TiledConfig& config, SuperPointFeatures& features)
    {
        if (!network.isReady()) {
            std::cerr << "TiledExtractor: the SuperPoint must be initialized with the tile size first." << std::endl;
            return false;
        }

        const ExtractFunction extractTile = [&](const unsigned char* tile, SuperPointFeatures& f) {
            return network.extract(tile, f);
            };

        return extract(extractTile, network.getWidth(), network.getHeight(), rgba, imageWidth, imageHeight, config, features);
    }
}
