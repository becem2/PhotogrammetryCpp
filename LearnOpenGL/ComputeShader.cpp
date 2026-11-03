
#include "ComputeShader.h"

#include <iostream>
#include <fstream>
#include <sstream>


ComputeShader::ComputeShader(const std::string& path)
{
    program = 0;

    const std::string& shaderPath = path;
    std::ifstream shaderFile(shaderPath);

    if (!shaderFile.is_open())
    {
        std::cerr << "Failed to open shader file: " << shaderPath << std::endl;
        return;
    }

    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    shaderFile.close();

    std::string shaderSource = shaderStream.str();

    if (shaderSource.empty())
    {
        std::cerr << "Shader file is empty: " << shaderPath << std::endl;
        return;
    }

    const char* shaderCode = shaderSource.c_str();

    GLint ok;
    char log[512];

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    if (shader == 0)
    {
        std::cerr << "Failed to create compute shader!" << std::endl;
        return;
    }

    glShaderSource(shader, 1, &shaderCode, NULL);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

    if (!ok)
    {
        glGetShaderInfoLog(shader, 512, NULL, log);
        std::cerr << "Compute shader compilation failed: " << log << std::endl;

        glDeleteShader(shader);
        return;
    }

    program = glCreateProgram();

    if (program == 0)
    {
        std::cerr << "Failed to create shader program!" << std::endl;

        glDeleteShader(shader);
        return;
    }

    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &ok);

    if (!ok)
    {
        glGetProgramInfoLog(program, 512, NULL, log);
        std::cerr << "Compute shader program linking failed: " << log << std::endl;

        glDeleteProgram(program);
        program = 0;

        glDeleteShader(shader);
        return;
    }

    glDetachShader(program, shader);
    glDeleteShader(shader);

}


GLuint ComputeShader::getProgram() const
{
    return program;
}


void ComputeShader::use() const
{
    if (program != 0)
    {
        glUseProgram(program);
    }
}


ComputeShader::~ComputeShader()
{
    if (program != 0)
    {
        glDeleteProgram(program);
    }
}
