
#pragma once

#include <glad.h>
#include <string>


class ComputeShader
{
private:

    GLuint program;

public:

    ComputeShader(const std::string& path);

    ComputeShader(const ComputeShader&) = delete;
    ComputeShader& operator=(const ComputeShader&) = delete;

    GLuint getProgram() const;

    void use() const;

    ~ComputeShader();
};
