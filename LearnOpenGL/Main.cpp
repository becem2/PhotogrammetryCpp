#define NOMINMAX
#define GLFW_INCLUDE_NONE

#include <iostream>

#include "Settings.h"
#include "Window.h"
#include "Pipeline.h"
#include "CommandLine.h"
#include "Function.h"

int main(int argc, char** argv)
{
	CommandLine cli;

	if (!CommandLineParser::parseCommandLine(argc, argv, cli)) {
		return -1;
	}

	// If no CLI arguments provided (argc == 1), prompt user for mode selection
	if (argc == 1) {
		cli.mode = CommandLineParser::selectApplicationMode();
	}

	GLFWwindow* window = Window::createWindow();

	if (window == nullptr) {
		return -1;
	}

	Function::checkGPUMemory();


	// ---------Route to appropriate mode---------
	bool success = false;

	switch (cli.mode) {

		case ApplicationMode::FeatureDetection:
		{
			std::cout << "\n>>> Starting Feature Detection Mode <<<" << std::endl;
			success = Pipeline::runFeatureDetectionMode();
			break;
		}

		case ApplicationMode::PseudoLabels:
		{
			std::cout << "\n>>> Starting Pseudo-Labels Mode <<<" << std::endl;
			success = CommandLineParser::runPseudoLabels(cli);
			break;
		}

		case ApplicationMode::Training:
		{
			std::cout << "\n>>> Starting Training Mode <<<" << std::endl;
			success = Pipeline::runTrainingMode();
			break;
		}
	}


	// ---------Window---------
	Window::runRenderLoop(window);


	// ---------Cleanup---------
	Function::checkGPUMemory();

	return success ? 0 : -1;
}
