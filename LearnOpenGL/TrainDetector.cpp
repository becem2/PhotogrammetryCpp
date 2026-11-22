#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "TrainDetector.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <set>

#include "SuperPointTrainer.h"
#include "TrainingData.h"


namespace
{
    void printStats(const char* name, const TrainingStats& s)
    {
        std::cout << name << " loss " << std::fixed << std::setprecision(4) << s.loss
            << " | hit " << std::setprecision(1) << s.hit * 100.0f << "%"
            << " | false alarm " << std::setprecision(2) << s.falseAlarm * 100.0f << "%";
    }

    // Loads the crops of `files` (already augmented if asked) into `crops` and fills `batch`.
    bool loadBatch(const std::vector<std::filesystem::path>& files, size_t first, size_t count,
        bool augment, std::mt19937& rng,
        std::vector<LabeledCrop>& crops, std::vector<TrainingSample>& batch)
    {
        crops.assign(count, LabeledCrop());
        batch.clear();

        std::bernoulli_distribution coin(0.5);

        for (size_t i = 0; i < count; i++) {

            if (!TrainingData::load(files[first + i], crops[i])) return false;

            if (augment) {
                const bool flipX = coin(rng);
                const bool flipY = coin(rng);

                if (flipX || flipY) TrainingData::flip(crops[i], flipX, flipY);
            }

            TrainingData::updateRgba(crops[i]);

            TrainingSample sample;
            sample.rgba = crops[i].rgba.data();
            sample.labels = crops[i].labels.data();
            batch.push_back(sample);
        }

        return true;
    }
}


bool trainDetector(SuperPoint& network, const TrainDetectorOptions& options)
{
    // Only crops of exactly the network size can be used (the GPU buffers are sized for it)
    std::vector<std::filesystem::path> all;
    size_t skippedSize = 0;

    for (const auto& f : TrainingData::listSamples(options.dataFolder)) {
        int w = 0, h = 0;

        if (TrainingData::readSize(f, w, h) && w == network.getWidth() && h == network.getHeight()) {
            all.push_back(f);
        }
        else {
            skippedSize++;
        }
    }

    if (skippedSize > 0) {
        std::cout << "Skipped " << skippedSize << " crops whose size is not "
            << network.getWidth() << " x " << network.getHeight() << std::endl;
    }

    if (all.empty()) {
        std::cerr << "No usable .sample files in " << options.dataFolder
            << " (run Mode 2: pseudo-labels, with the same crop size)." << std::endl;
        return false;
    }

    // ---- split by IMAGE, like the Python script ----
    std::vector<std::string> stems;
    for (const auto& f : all) stems.push_back(TrainingData::imageStem(f));
    std::sort(stems.begin(), stems.end());
    stems.erase(std::unique(stems.begin(), stems.end()), stems.end());

    std::mt19937 rng(options.seed);
    std::shuffle(stems.begin(), stems.end(), rng);

    const size_t valImages = stems.size() >= 2
        ? std::max<size_t>(1, static_cast<size_t>(stems.size() * options.validationFraction + 0.5f)) : 0;

    const std::set<std::string> valStems(stems.begin(), stems.begin() + valImages);

    std::vector<std::filesystem::path> trainFiles, valFiles;
    for (const auto& f : all) {
        (valStems.count(TrainingData::imageStem(f)) ? valFiles : trainFiles).push_back(f);
    }

    std::cout << trainFiles.size() << " training crops, " << valFiles.size() << " validation crops" << std::endl;

    if (trainFiles.size() < static_cast<size_t>(options.batchSize)) {
        std::cerr << "Only " << trainFiles.size() << " training crops for a batch size of " << options.batchSize
            << ": lower the batch size or generate more pseudo-labels." << std::endl;
        return false;
    }

    // ---- trainer ----
    SuperPointTrainerConfig config;
    config.learningRate = options.learningRate;
    config.shaderDirectory = network.getConfig().shaderDirectory;

    SuperPointTrainer trainer(network, config);

    if (!trainer.initialize()) return false;

    std::vector<LabeledCrop> crops;
    std::vector<TrainingSample> batch;

    // Validation: crops are loaded in chunks (never all in memory)
    auto validate = [&](TrainingStats& result) -> bool
        {
            double loss = 0, hit = 0, alarm = 0;
            size_t chunks = 0;

            for (size_t first = 0; first < valFiles.size(); first += options.batchSize) {

                const size_t count = std::min<size_t>(options.batchSize, valFiles.size() - first);

                if (!loadBatch(valFiles, first, count, false, rng, crops, batch)) return false;

                TrainingStats s;
                if (!trainer.evaluate(batch, s)) return false;

                // equal-size chunks: average of the chunk means (last chunk weighs a bit more, fine for monitoring)
                loss += s.loss; hit += s.hit; alarm += s.falseAlarm;
                chunks++;
            }

            result.loss = static_cast<float>(loss / std::max<size_t>(chunks, 1));
            result.hit = static_cast<float>(hit / std::max<size_t>(chunks, 1));
            result.falseAlarm = static_cast<float>(alarm / std::max<size_t>(chunks, 1));

            return true;
        };

    float best = 1e30f;

    if (!valFiles.empty()) {
        TrainingStats base;
        if (!validate(base)) return false;

        std::cout << "epoch  0 (starting weights) | "; printStats("val  ", base); std::cout << std::endl;
        best = base.loss;   // an epoch must BEAT the starting weights to be saved
    }

    int badEpochs = 0;
    bool saved = false;

    std::ofstream log(options.outputWeights + ".csv");
    log << "epoch,train_loss,train_hit,train_false_alarm,val_loss,val_hit,val_false_alarm,seconds\n";

    for (int epoch = 1; epoch <= options.epochs; epoch++) {

        const auto t0 = std::chrono::steady_clock::now();

        std::shuffle(trainFiles.begin(), trainFiles.end(), rng);

        double loss = 0, hit = 0, alarm = 0;
        size_t steps = 0;

        // drop_last = True, like the Python DataLoader
        for (size_t first = 0; first + options.batchSize <= trainFiles.size(); first += options.batchSize) {

            if (!loadBatch(trainFiles, first, options.batchSize, true, rng, crops, batch)) return false;

            TrainingStats s;
            if (!trainer.trainStep(batch, s)) return false;

            loss += s.loss; hit += s.hit; alarm += s.falseAlarm;
            steps++;

            if (steps % 50 == 0) {
                std::cout << "  epoch " << epoch << " step " << steps << " loss " << s.loss << std::endl;
            }
        }

        TrainingStats train;
        train.loss = static_cast<float>(loss / std::max<size_t>(steps, 1));
        train.hit = static_cast<float>(hit / std::max<size_t>(steps, 1));
        train.falseAlarm = static_cast<float>(alarm / std::max<size_t>(steps, 1));

        TrainingStats val;
        float score = train.loss;

        std::cout << "epoch " << std::setw(2) << epoch << " | "; printStats("train", train);

        if (!valFiles.empty()) {
            if (!validate(val)) return false;

            std::cout << "\n         | "; printStats("val  ", val);
            score = val.loss;
        }

        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::cout << "   (" << std::fixed << std::setprecision(0) << seconds << " s)" << std::endl;

        log << epoch << "," << train.loss << "," << train.hit << "," << train.falseAlarm << ","
            << val.loss << "," << val.hit << "," << val.falseAlarm << "," << seconds << "\n";
        log.flush();

        if (score < best) {
            best = score;
            badEpochs = 0;
            saved = true;

            if (trainer.saveWeights(options.outputWeights)) {
                std::cout << "         -> saved best weights to " << options.outputWeights << std::endl;
            }
        }
        else {
            badEpochs++;

            if (options.patience > 0 && badEpochs >= options.patience) {
                std::cout << "No improvement for " << options.patience << " epochs: stopping early." << std::endl;
                break;
            }
        }
    }

    if (!saved) {
        std::cout << "No epoch beat the starting weights, so no weights file was written." << std::endl;
    }

    return true;
}
