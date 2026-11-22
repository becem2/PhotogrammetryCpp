#pragma once

#include <glad.h>
#include <glfw3.h>

namespace Window
{
	// Window state
	extern int screenWidth;
	extern int screenHeight;

	// Timing
	extern float deltaTime;
	extern float lastFrame;

	// Cursor mode
	enum class CursorMode { Captured, Free };
	extern CursorMode cursorMode;

	// Window creation and management
	GLFWwindow* createWindow();
	void runRenderLoop(GLFWwindow* window);

	// Input callbacks
	void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void processInput(GLFWwindow* window);
}
