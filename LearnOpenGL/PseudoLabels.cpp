#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "PseudoLabels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Function.h"
#include "stb_image.h"


namespace
{
    using Mat3 = std::array<double, 9>;      // row-major 3x3

    const Mat3 kIdentity = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };


    //-----------Homography Maths---------//

    bool invert(const Mat3& m, Mat3& out)
    {
        const double a = m[0], b = m[1], c = m[2];
        const double d = m[3], e = m[4], f = m[5];
        const double g = m[6], h = m[7], i = m[8];

        const double A = e * i - f * h;
        const double B = -(d * i - f * g);
        const double C = d * h - e * g;

        const double det = a * A + b * B + c * C;

        if (std::abs(det) < 1e-12) return false;

        const double r = 1.0 / det;

        out = { A * r,                -(b * i - c * h) * r,  (b * f - c * e) * r,
                B * r,                 (a * i - c * g) * r, -(a * f - c * d) * r,
                C * r,                -(a * h - b * g) * r,  (a * e - b * d) * r };

        return true;
    }


    // 4 point pairs -> homography with h33 = 1 (same as cv2.getPerspectiveTransform)
    bool homographyFromCorners(const double src[4][2], const double dst[4][2], Mat3& out)
    {
        double A[8][9] = {};

        for (int k = 0; k < 4; k++) {
            const double x = src[k][0], y = src[k][1];
            const double u = dst[k][0], v = dst[k][1];

            const double r0[9] = { x, y, 1, 0, 0, 0, -x * u, -y * u, u };
            const double r1[9] = { 0, 0, 0, x, y, 1, -x * v, -y * v, v };

            std::copy(r0, r0 + 9, A[2 * k]);
            std::copy(r1, r1 + 9, A[2 * k + 1]);
        }

        for (int col = 0; col < 8; col++) {

            int pivot = col;
            for (int r = col + 1; r < 8; r++) {
                if (std::abs(A[r][col]) > std::abs(A[pivot][col])) pivot = r;
            }

            if (std::abs(A[pivot][col]) < 1e-12) return false;

            std::swap_ranges(A[col], A[col] + 9, A[pivot]);

            for (int r = 0; r < 8; r++) {
                if (r == col) continue;

                const double factor = A[r][col] / A[col][col];

                for (int k = col; k < 9; k++) A[r][k] -= factor * A[col][k];
            }
        }

        for (int k = 0; k < 8; k++) out[k] = A[k][8] / A[k][k];
        out[8] = 1.0;

        return true;
    }


    // Same construction as sample_homography() in MakePseudoLabels.py:
    // move the 4 corners (perspective noise, scale, rotation around the centre, translation).
    Mat3 sampleHomography(int w, int h, std::mt19937_64& rng, const PseudoLabelOptions& o)
    {
        const double W = w, H = h;
        const double cx = W / 2.0, cy = H / 2.0;

        const double src[4][2] = { {0, 0}, {W, 0}, {W, H}, {0, H} };

        std::normal_distribution<double> noise(0.0, o.perspective / 2.0);
        std::uniform_real_distribution<double> scaleDist(1.0 - o.scaleAmplitude, 1.0 + o.scaleAmplitude);
        std::uniform_real_distribution<double> angleDist(-o.maxAngle, o.maxAngle);
        std::uniform_real_distribution<double> shiftDist(-o.translation, o.translation);

        double pts[4][2];

        for (int k = 0; k < 4; k++) {
            pts[k][0] = src[k][0] - cx;
            pts[k][1] = src[k][1] - cy;
        }

        for (int k = 0; k < 4; k++) {
            pts[k][0] += std::clamp(noise(rng), -double(o.perspective), double(o.perspective)) * W;
            pts[k][1] += std::clamp(noise(rng), -double(o.perspective), double(o.perspective)) * H;
        }

        const double s = scaleDist(rng);
        const double a = angleDist(rng);
        const double ca = std::cos(a), sa = std::sin(a);

        const double shiftX = shiftDist(rng) * W;
        const double shiftY = shiftDist(rng) * H;

        double dst[4][2];

        for (int k = 0; k < 4; k++) {
            const double x = pts[k][0] * s;
            const double y = pts[k][1] * s;

            dst[k][0] = ca * x - sa * y + cx + shiftX;
            dst[k][1] = sa * x + ca * y + cy + shiftY;
        }

        Mat3 result;

        if (!homographyFromCorners(src, dst, result)) return kIdentity;

        return result;
    }


    //-----------Warping---------//

    // dst(x, y) = src(M * (x, y)): M maps DESTINATION pixels to SOURCE pixels.
    // Bilinear, pixels outside the source are 0 (same as cv2.warpPerspective with a constant border).
    void warpBilinear(const std::vector<float>& src, int w, int h, const Mat3& M, std::vector<float>& dst)
    {
        dst.assign(src.size(), 0.0f);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                const double dw = M[6] * x + M[7] * y + M[8];
                if (std::abs(dw) < 1e-12) continue;

                const double sx = (M[0] * x + M[1] * y + M[2]) / dw;
                const double sy = (M[3] * x + M[4] * y + M[5]) / dw;

                if (sx < -1.0 || sx > w || sy < -1.0 || sy > h) continue;

                const int x0 = static_cast<int>(std::floor(sx));
                const int y0 = static_cast<int>(std::floor(sy));
                const float fx = static_cast<float>(sx - x0);
                const float fy = static_cast<float>(sy - y0);

                auto at = [&](int xx, int yy) -> float
                    {
                        return (xx < 0 || yy < 0 || xx >= w || yy >= h) ? 0.0f : src[static_cast<size_t>(yy) * w + xx];
                    };

                dst[static_cast<size_t>(y) * w + x] =
                    (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) +
                    (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
            }
        }
    }


    // 1 where the warped image has real content (nearest neighbour), 0 elsewhere
    void warpMask(int w, int h, const Mat3& M, std::vector<unsigned char>& mask)
    {
        mask.assign(static_cast<size_t>(w) * h, 0);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                const double dw = M[6] * x + M[7] * y + M[8];
                if (std::abs(dw) < 1e-12) continue;

                const double sx = (M[0] * x + M[1] * y + M[2]) / dw;
                const double sy = (M[3] * x + M[4] * y + M[5]) / dw;

                if (sx < -1.0 || sx > w || sy < -1.0 || sy > h) continue;

                const int ix = static_cast<int>(std::floor(sx + 0.5));
                const int iy = static_cast<int>(std::floor(sy + 0.5));

                if (ix >= 0 && ix < w && iy >= 0 && iy < h) mask[static_cast<size_t>(y) * w + x] = 1;
            }
        }
    }


    // Square erosion of a 0/1 mask, O(pixels) through prefix counts of the zeros.
    // Outside the picture does not count as zero (like OpenCV's default erode border).
    void erodeMask(std::vector<unsigned char>& mask, int w, int h, int radius)
    {
        if (radius <= 0) return;

        std::vector<unsigned char> tmp(mask.size());
        std::vector<int> prefix(static_cast<size_t>(std::max(w, h)) + 1);

        // rows
        for (int y = 0; y < h; y++) {
            prefix[0] = 0;
            for (int x = 0; x < w; x++) prefix[x + 1] = prefix[x] + (mask[static_cast<size_t>(y) * w + x] == 0);

            for (int x = 0; x < w; x++) {
                const int lo = std::max(0, x - radius), hi = std::min(w - 1, x + radius);
                tmp[static_cast<size_t>(y) * w + x] = (prefix[hi + 1] - prefix[lo]) == 0 ? 1 : 0;
            }
        }

        // columns
        for (int x = 0; x < w; x++) {
            prefix[0] = 0;
            for (int y = 0; y < h; y++) prefix[y + 1] = prefix[y] + (tmp[static_cast<size_t>(y) * w + x] == 0);

            for (int y = 0; y < h; y++) {
                const int lo = std::max(0, y - radius), hi = std::min(h - 1, y + radius);
                mask[static_cast<size_t>(y) * w + x] = (prefix[hi + 1] - prefix[lo]) == 0 ? 1 : 0;
            }
        }
    }


    //-----------Images---------//

    // 24-bit BMP of the crop with a red square on every pseudo-label keypoint.
    bool saveLabelPicture(const std::string& path, const std::vector<unsigned char>& gray,
        int w, int h, const std::vector<LabelPoint>& points)
    {
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);

        for (size_t i = 0; i < gray.size(); i++) {
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray[i];
        }

        for (const LabelPoint& p : points) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    const int x = p.x + dx, y = p.y + dy;

                    if (x < 0 || y < 0 || x >= w || y >= h) continue;

                    unsigned char* px = &rgb[(static_cast<size_t>(y) * w + x) * 3];
                    px[0] = 255; px[1] = 0; px[2] = 0;
                }
            }
        }

        const int rowBytes = (w * 3 + 3) & ~3;
        const uint32_t dataSize = static_cast<uint32_t>(rowBytes) * h;
        const uint32_t fileSize = 54 + dataSize;

        std::ofstream file(path, std::ios::binary);

        if (!file) return false;

        auto put32 = [&](uint32_t v) { file.write(reinterpret_cast<const char*>(&v), 4); };
        auto put16 = [&](uint16_t v) { file.write(reinterpret_cast<const char*>(&v), 2); };

        file.write("BM", 2);
        put32(fileSize); put32(0); put32(54);
        put32(40); put32(static_cast<uint32_t>(w)); put32(static_cast<uint32_t>(h));
        put16(1); put16(24); put32(0); put32(dataSize); put32(2835); put32(2835); put32(0); put32(0);

        std::vector<unsigned char> row(rowBytes, 0);

        for (int y = h - 1; y >= 0; y--) {          // BMP is stored bottom-up, BGR
            for (int x = 0; x < w; x++) {
                const unsigned char* px = &rgb[(static_cast<size_t>(y) * w + x) * 3];
                row[x * 3] = px[2]; row[x * 3 + 1] = px[1]; row[x * 3 + 2] = px[0];
            }
            file.write(reinterpret_cast<const char*>(row.data()), rowBytes);
        }

        return static_cast<bool>(file);
    }
}


//-----------Homographic Adaptation---------//

bool homographicAdaptation(const std::vector<unsigned char>& gray, int width, int height,
    const HeatmapFunction& detect, const PseudoLabelOptions& options, std::mt19937_64& rng,
    std::vector<float>& aggregated, std::vector<float>& single)
{
    const size_t n = static_cast<size_t>(width) * height;
    const int views = std::max(options.homographies, 1);

    std::vector<float> image(n);
    for (size_t i = 0; i < n; i++) image[i] = gray[i];

    std::vector<float> sumHeat(n, 0.0f), counts(n, 0.0f);
    std::vector<float> warped, heat, masked, backHeat, backValid, valid(n);
    std::vector<unsigned char> mask, rgba(n * 4);

    single.clear();

    for (int v = 0; v < views; v++) {

        const Mat3 H = (v == 0) ? kIdentity : sampleHomography(width, height, rng, options);

        Mat3 Hinv;
        if (!invert(H, Hinv)) continue;

        // Warped view -> RGBA8 for the network (gray rounded to 8 bits, like the Python uint8 warp)
        warpBilinear(image, width, height, Hinv, warped);

        for (size_t i = 0; i < n; i++) {
            const unsigned char g = static_cast<unsigned char>(std::clamp(warped[i] + 0.5f, 0.0f, 255.0f));
            rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = g;
            rgba[i * 4 + 3] = 255;
        }

        // Where the warped view really has image content (shrunk so the black border cannot fool the detector)
        warpMask(width, height, Hinv, mask);
        erodeMask(mask, width, height, options.erode);

        for (size_t i = 0; i < n; i++) valid[i] = mask[i];

        if (!detect(rgba.data(), heat) || heat.size() != n) {
            std::cerr << "homographicAdaptation: the detector failed on view " << v << std::endl;
            return false;
        }

        if (v == 0) single = heat;

        masked.resize(n);
        for (size_t i = 0; i < n; i++) masked[i] = heat[i] * valid[i];

        // Back to the original view
        warpBilinear(masked, width, height, H, backHeat);
        warpBilinear(valid, width, height, H, backValid);

        for (size_t i = 0; i < n; i++) {
            sumHeat[i] += backHeat[i];
            counts[i] += backValid[i];
        }
    }

    aggregated.assign(n, 0.0f);

    const float minCount = options.minValid * static_cast<float>(views);

    for (size_t i = 0; i < n; i++) {
        aggregated[i] = (counts[i] < minCount) ? 0.0f : sumHeat[i] / std::max(counts[i], 1e-6f);
    }

    if (single.empty()) single.assign(n, 0.0f);

    return true;
}


//-----------Keypoints---------//

std::vector<LabelPoint> detectKeypoints(const std::vector<float>& heatmap, int width, int height,
    float threshold, int nmsRadius, int borderSize)
{
    std::vector<LabelPoint> candidates;

    for (int y = borderSize; y < height - borderSize; y++) {
        for (int x = borderSize; x < width - borderSize; x++) {

            const float s = heatmap[static_cast<size_t>(y) * width + x];

            if (s > threshold) candidates.push_back({ x, y, s });
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(),
        [](const LabelPoint& a, const LabelPoint& b) { return a.score > b.score; });

    std::vector<unsigned char> suppressed(static_cast<size_t>(width) * height, 0);
    std::vector<LabelPoint> keep;

    for (const LabelPoint& c : candidates) {

        if (suppressed[static_cast<size_t>(c.y) * width + c.x]) continue;

        keep.push_back(c);

        for (int dy = -nmsRadius; dy <= nmsRadius; dy++) {
            for (int dx = -nmsRadius; dx <= nmsRadius; dx++) {
                const int x = c.x + dx, y = c.y + dy;

                if (x >= 0 && x < width && y >= 0 && y < height) suppressed[static_cast<size_t>(y) * width + x] = 1;
            }
        }
    }

    return keep;
}


//-----------Whole Stage---------//

bool makePseudoLabels(SuperPoint& network, const PseudoLabelOptions& options)
{
    if (!network.isReady()) {
        std::cerr << "makePseudoLabels: the SuperPoint must be initialized first." << std::endl;
        return false;
    }

    const int cropW = network.getWidth();
    const int cropH = network.getHeight();

    std::vector<std::string> paths;
    int totalFiles = 0;

    if (!Function::collectImagePaths(options.inputFolder, paths, totalFiles)) return false;

    if (paths.empty()) {
        std::cerr << "makePseudoLabels: no images in " << options.inputFolder << std::endl;
        return false;
    }

    if (!Function::createDirectory(options.outputFolder)) return false;

    const int first = std::max(options.startIndex, 0);
    const int last = (options.endIndex < 0) ? static_cast<int>(paths.size())
        : std::min(options.endIndex, static_cast<int>(paths.size()));

    if (first >= last) {
        std::cerr << "makePseudoLabels: empty image range " << first << " to " << last
            << " (the folder has " << paths.size() << " images)." << std::endl;
        return false;
    }

    std::cout << "Images " << first << " to " << last - 1 << " of " << paths.size() << " x "
        << options.cropsPerImage << " crops x " << options.homographies << " views, crop "
        << cropW << "x" << cropH << std::endl;

    std::mt19937_64 rng(options.seed);

    const HeatmapFunction detect = [&](const unsigned char* rgba, std::vector<float>& heat)
        {
            return network.computeHeatmap(rgba, heat);
        };

    const std::filesystem::path outDir(options.outputFolder);

    for (int i = first; i < last; i++) {

        const std::filesystem::path imagePath(paths[i]);
        const std::string stem = imagePath.stem().string();

        if (options.resume) {
            bool allDone = true;

            for (int k = 0; k < options.cropsPerImage; k++) {
                if (!std::filesystem::exists(outDir / (stem + "_c" + std::to_string(k) + ".sample"))) allDone = false;
            }

            if (allDone) {
                std::cout << "[" << i << "] " << imagePath.filename().string() << ": already done, skipped" << std::endl;
                continue;
            }
        }

        int w = 0, h = 0, channels = 0;
        unsigned char* pixels = stbi_load(paths[i].c_str(), &w, &h, &channels, 1);

        if (pixels == nullptr) {
            std::cerr << "[" << i << "] cannot read " << paths[i] << std::endl;
            continue;
        }

        if (w < cropW || h < cropH) {
            std::cerr << "[" << i << "] " << imagePath.filename().string() << ": " << w << "x" << h
                << " is smaller than the crop " << cropW << "x" << cropH << ", skipped" << std::endl;
            stbi_image_free(pixels);
            continue;
        }

        for (int k = 0; k < options.cropsPerImage; k++) {

            const auto t0 = std::chrono::steady_clock::now();

            const int ox = static_cast<int>(rng() % static_cast<uint64_t>(w - cropW + 1));
            const int oy = static_cast<int>(rng() % static_cast<uint64_t>(h - cropH + 1));

            std::vector<unsigned char> crop(static_cast<size_t>(cropW) * cropH);

            for (int y = 0; y < cropH; y++) {
                std::copy(pixels + (static_cast<size_t>(oy + y) * w + ox),
                    pixels + (static_cast<size_t>(oy + y) * w + ox + cropW),
                    crop.begin() + static_cast<size_t>(y) * cropW);
            }

            std::vector<float> aggregated, single;

            if (!homographicAdaptation(crop, cropW, cropH, detect, options, rng, aggregated, single)) {
                stbi_image_free(pixels);
                return false;
            }

            const std::vector<LabelPoint> points = detectKeypoints(aggregated, cropW, cropH,
                options.detectionThreshold, options.nmsRadius, options.borderSize);

            const size_t singleCount = detectKeypoints(single, cropW, cropH,
                options.detectionThreshold, options.nmsRadius, options.borderSize).size();

            const std::string name = stem + "_c" + std::to_string(k);

            const std::vector<int32_t> labels = TrainingData::makeLabelMap(points, cropW, cropH);

            if (!TrainingData::save(outDir / (name + ".sample"), cropW, cropH, crop, labels)) {
                std::cerr << "cannot write " << (outDir / (name + ".sample")).string() << std::endl;
                stbi_image_free(pixels);
                return false;
            }

            if (options.saveVisualization) {
                saveLabelPicture((outDir / (name + "_labels.bmp")).string(), crop, cropW, cropH, points);
            }

            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            std::cout << "[" << i << "] " << name << ": single pass " << singleCount
                << " keypoints -> adapted " << points.size() << " keypoints  (" << seconds << " s)" << std::endl;
        }

        stbi_image_free(pixels);
    }

    std::cout << "Done. Labels saved to " << options.outputFolder << std::endl;

    return true;
}
