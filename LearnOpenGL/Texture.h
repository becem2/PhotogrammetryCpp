
#pragma once

#include <glad.h>
#include <iostream>


enum class LayerType {
    NORMAL,
    CONV,
    RELU,
    MAXPOOL
};


class Texture {
private:

    GLuint textureID;
    GLenum internalFormat;
    LayerType layerType;
    bool isArray;
    unsigned int Height, Width, ConvLayers;

public:

    Texture(LayerType layerType, bool isArray, GLenum internalFormat, unsigned int Height, unsigned int Width, unsigned int ConvLayers);

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    GLuint getTextureID() const;

    LayerType getLayerType() const;

    void resize(unsigned int Height, unsigned int Width, unsigned int ConvLayers);

    ~Texture();
};
