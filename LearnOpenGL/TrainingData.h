#pragma once

// Helpers to read the .sample crops written by makePseudoLabels() (Mode 2),
// flip them (labels included) and turn them into RGBA for SuperPoint.
//
// .sample binary layout (little endian):
//   int32 height, int32 width, uint8 gray[height * width], int32 labels[(height / 8) * (width / 8)]
// labels: 0..63 = keypoint position inside the 8x8 cell (row * 8 + column), 64 = no keypoint.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// A pseudo-label keypoint (pixel position + confidence)
struct LabelPoint
{
    int x = 0;
    int y = 0;
    float score = 0.0f;
};


struct LabeledCrop
{
    int width = 0;
    int height = 0;

    std::vector<unsigned char> gray;      // width * height
    std::vector<int32_t> labels;          // (height / 8) * (width / 8)
    std::vector<unsigned char> rgba;      // width * height * 4, rebuilt by updateRgba()
};


namespace TrainingData
{
    inline bool load(const std::filesystem::path& path, LabeledCrop& crop)
    {
        std::ifstream file(path, std::ios::binary);

        int32_t size[2] = {};
        file.read(reinterpret_cast<char*>(size), sizeof(size));

        if (!file || size[0] <= 0 || size[1] <= 0 || size[0] % 8 != 0 || size[1] % 8 != 0) {
            std::cerr << "TrainingData: bad header in " << path.string() << std::endl;
            return false;
        }

        crop.height = size[0];
        crop.width = size[1];
        crop.gray.resize(static_cast<size_t>(crop.width) * crop.height);
        crop.labels.resize(static_cast<size_t>(crop.width / 8) * (crop.height / 8));

        file.read(reinterpret_cast<char*>(crop.gray.data()), static_cast<std::streamsize>(crop.gray.size()));
        file.read(reinterpret_cast<char*>(crop.labels.data()),
            static_cast<std::streamsize>(crop.labels.size() * sizeof(int32_t)));

        if (!file) {
            std::cerr << "TrainingData: file too short: " << path.string() << std::endl;
            return false;
        }

        return true;
    }


    // Reads only the 8-byte header of a .sample file.
    inline bool readSize(const std::filesystem::path& path, int& width, int& height)
    {
        std::ifstream file(path, std::ios::binary);

        int32_t size[2] = {};
        file.read(reinterpret_cast<char*>(size), sizeof(size));

        if (!file || size[0] <= 0 || size[1] <= 0) return false;

        height = size[0];
        width = size[1];

        return true;
    }


    // A flip mirrors the cell grid AND the position inside every cell.
    inline int32_t flipLabel(int32_t label, bool flipX, bool flipY)
    {
        if (label < 0 || label >= 64) return 64;

        int row = label / 8;
        int col = label % 8;

        if (flipX) col = 7 - col;
        if (flipY) row = 7 - row;

        return row * 8 + col;
    }


    inline void flip(LabeledCrop& crop, bool flipX, bool flipY)
    {
        const int w = crop.width;
        const int h = crop.height;
        const int w3 = w / 8;
        const int h3 = h / 8;

        std::vector<unsigned char> gray(crop.gray.size());
        std::vector<int32_t> labels(crop.labels.size());

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const int sx = flipX ? (w - 1 - x) : x;
                const int sy = flipY ? (h - 1 - y) : y;
                gray[static_cast<size_t>(y) * w + x] = crop.gray[static_cast<size_t>(sy) * w + sx];
            }
        }

        for (int cy = 0; cy < h3; cy++) {
            for (int cx = 0; cx < w3; cx++) {
                const int sx = flipX ? (w3 - 1 - cx) : cx;
                const int sy = flipY ? (h3 - 1 - cy) : cy;
                labels[static_cast<size_t>(cy) * w3 + cx] =
                    flipLabel(crop.labels[static_cast<size_t>(sy) * w3 + sx], flipX, flipY);
            }
        }

        crop.gray.swap(gray);
        crop.labels.swap(labels);
    }


    // SuperPoint::extract/encode take RGBA8 (its first shader converts to gray).
    inline void updateRgba(LabeledCrop& crop)
    {
        crop.rgba.resize(crop.gray.size() * 4);

        for (size_t i = 0; i < crop.gray.size(); i++) {
            crop.rgba[i * 4 + 0] = crop.gray[i];
            crop.rgba[i * 4 + 1] = crop.gray[i];
            crop.rgba[i * 4 + 2] = crop.gray[i];
            crop.rgba[i * 4 + 3] = 255;
        }
    }


    // Image name of a crop "<image>_c<k>.sample" -> "<image>" (split train / validation by IMAGE)
    inline std::string imageStem(const std::filesystem::path& file)
    {
        const std::string stem = file.stem().string();
        const size_t pos = stem.rfind("_c");

        return pos == std::string::npos ? stem : stem.substr(0, pos);
    }


    inline std::vector<std::filesystem::path> listSamples(const std::string& folder)
    {
        std::vector<std::filesystem::path> files;

        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (entry.is_regular_file() && entry.path().extension() == ".sample") {
                files.push_back(entry.path());
            }
        }

        std::sort(files.begin(), files.end());

        return files;
    }

    // Keypoints -> one label per 8x8 cell (port of make_label_map in Traindetector.py).
    // If two keypoints share a cell, the one with the higher score is kept.
    inline std::vector<int32_t> makeLabelMap(std::vector<LabelPoint> points, int width, int height)
    {
        std::vector<int32_t> labels(static_cast<size_t>(width / 8) * (height / 8), 64);

        std::stable_sort(points.begin(), points.end(),
            [](const LabelPoint& a, const LabelPoint& b) { return a.score < b.score; });   // worst first, best written last

        for (const LabelPoint& p : points) {
            if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height) {
                labels[static_cast<size_t>(p.y / 8) * (width / 8) + (p.x / 8)] = (p.y % 8) * 8 + (p.x % 8);
            }
        }

        return labels;
    }


    inline bool save(const std::filesystem::path& path, int width, int height,
        const std::vector<unsigned char>& gray, const std::vector<int32_t>& labels)
    {
        std::ofstream file(path, std::ios::binary);

        const int32_t size[2] = { height, width };

        file.write(reinterpret_cast<const char*>(size), sizeof(size));
        file.write(reinterpret_cast<const char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
        file.write(reinterpret_cast<const char*>(labels.data()),
            static_cast<std::streamsize>(labels.size() * sizeof(int32_t)));

        return static_cast<bool>(file);
    }
}
