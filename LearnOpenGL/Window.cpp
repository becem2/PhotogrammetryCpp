#include "Window.h"

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <iostream>

namespace Window
{
	// Window state
	int screenWidth = 0;
	int screenHeight = 0;

	// Timing
	float deltaTime = 0.0f;
	float lastFrame = 0.0f;

	// Cursor mode
	CursorMode cursorMode = CursorMode::Free;

	// Input callbacks
	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void mouse_callback(GLFWwindow* window, double xpos, double ypos)
	{
		if (cursorMode != CursorMode::Captured) {
			return;
		}

		// Camera::mouse_callback(window, xpos, ypos);
	}

	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
	{
		// Camera::scroll_callback(window, xoffset, yoffset);
	}

	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
	{
		ImGuiIO& io = ImGui::GetIO();

		if (io.WantCaptureMouse) {
			return;
		}

		if (button == GLFW_MOUSE_BUTTON_LEFT &&
			action == GLFW_PRESS &&
			cursorMode == CursorMode::Free) {

			cursorMode = CursorMode::Captured;

			glfwSetInputMode(
				window,
				GLFW_CURSOR,
				GLFW_CURSOR_DISABLED
			);
		}
	}

	void processInput(GLFWwindow* window)
	{
		static bool escWasPressed = false;

		bool escPressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;

		if (escPressed && !escWasPressed) {

			if (cursorMode == CursorMode::Captured) {

				cursorMode = CursorMode::Free;

				glfwSetInputMode(
					window,
					GLFW_CURSOR,
					GLFW_CURSOR_NORMAL
				);
			}
			else {

				glfwSetWindowShouldClose(window, true);
			}
		}

		escWasPressed = escPressed;

		if (cursorMode != CursorMode::Captured) {
			return;
		}

		/*
		glm::vec3 currentPos = ourCamera->getPosition();
		glm::vec3 front = ourCamera->getFront();
		glm::vec3 right = glm::normalize(glm::cross(front, ourCamera->getUp()));

		float cameraSpeed = 2.5f * deltaTime;

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			ourCamera->setPosition(currentPos.x + front.x * cameraSpeed,
				currentPos.y + front.y * cameraSpeed,
				currentPos.z + front.z * cameraSpeed);
		*/
	}

	GLFWwindow* createWindow()
	{
		if (!glfwInit()) {
			std::cout << "Failed to initialize GLFW" << std::endl;
			return nullptr;
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

		screenWidth = 800;
		screenHeight = 600;

		GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "LearnOpenGL", NULL, NULL);

		if (window == NULL) {
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return nullptr;
		}

		glfwMakeContextCurrent(window);

		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);
		glfwSetMouseButtonCallback(window, mouse_button_callback);

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		// glad: load all OpenGL function pointers
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			std::cout << "Failed to initialize GLAD" << std::endl;
			glfwDestroyWindow(window);
			glfwTerminate();
			return nullptr;
		}

		// configure global OpenGL state
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		glEnable(GL_STENCIL_TEST);
		glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

		return window;
	}

	void runRenderLoop(GLFWwindow* window)
	{
		IMGUI_CHECKVERSION();

		ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO();
		(void)io;

		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 430");

		while (!glfwWindowShouldClose(window)) {

			// ImGui frame start

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();


			// per-frame time logic

			float currentFrame = static_cast<float>(glfwGetTime());

			deltaTime = currentFrame - lastFrame;
			lastFrame = currentFrame;


			// input

			processInput(window);


			// render

			glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

			glClear(
				GL_COLOR_BUFFER_BIT |
				GL_DEPTH_BUFFER_BIT |
				GL_STENCIL_BUFFER_BIT
			);


			// ImGui rendering

			ImGui::Render();

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();

		ImGui::DestroyContext();

		glfwDestroyWindow(window);
		glfwTerminate();
	}
}
