
#include "Texture.h"


Texture::Texture(LayerType layerType, bool isArray, GLenum internalFormat, unsigned int Height, unsigned int Width, unsigned int ConvLayers) {

    this->layerType = layerType;
    this->isArray = isArray;
    this->Height = Height;
    this->Width = Width;
    this->ConvLayers = ConvLayers;
    this->internalFormat = internalFormat;

    textureID = 0;

    if (Width == 0 || Height == 0 || (isArray && ConvLayers == 0)) {
        std::cerr << "Invalid texture dimensions!" << std::endl;
        return;
    }

    glGenTextures(1, &textureID);

    if (textureID == 0) {
        std::cerr << "Failed to generate texture!" << std::endl;
        return;
    }

    if (isArray) {

        GLenum format = GL_RED;
        GLenum type = GL_FLOAT;

        if (internalFormat == GL_RGBA8) {
            format = GL_RGBA;
            type = GL_UNSIGNED_BYTE;
        }
        else if (internalFormat == GL_R8) {
            format = GL_RED;
            type = GL_UNSIGNED_BYTE;
        }

        glBindTexture(GL_TEXTURE_2D_ARRAY, textureID);

        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, internalFormat, Width, Height, ConvLayers, 0, format, type, NULL);

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }
    else {

        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, Width, Height);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

}


GLuint Texture::getTextureID() const {
    return textureID;
}


LayerType Texture::getLayerType() const {
    return layerType;
}


void Texture::resize(unsigned int Height, unsigned int Width, unsigned int ConvLayers) {

    if (!isArray) {
        std::cerr << "Cannot resize a non-array texture using this method!" << std::endl;
        return;
    }

    if (Width == 0 || Height == 0 || ConvLayers == 0) {
        std::cerr << "Invalid texture dimensions!" << std::endl;
        return;
    }

    if (this->Width == Width && this->Height == Height && this->ConvLayers == ConvLayers) {
        return;
    }

    GLenum format = GL_RED;
    GLenum type = GL_FLOAT;

    if (internalFormat == GL_RGBA8) {
        format = GL_RGBA;
        type = GL_UNSIGNED_BYTE;
    }
    else if (internalFormat == GL_R8) {
        format = GL_RED;
        type = GL_UNSIGNED_BYTE;
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, textureID);

    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, internalFormat, Width, Height, ConvLayers, 0, format, type, NULL);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    this->Height = Height;
    this->Width = Width;
    this->ConvLayers = ConvLayers;

}


Texture::~Texture() {
    if (textureID != 0) {
        glDeleteTextures(1, &textureID);
    }
}
