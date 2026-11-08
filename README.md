# Photogrammetry Pipeline

<p align="center">
  <strong>A from-scratch photogrammetry pipeline implemented in C++, OpenGL and GLSL.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Progress-5%2F12%20Steps-orange?style=flat-square" />
</p>

## About

This is a personal project aimed at recreating a fully functional photogrammetry pipeline from scratch, implementing its core algorithms in **C++, OpenGL and GLSL**.

The project follows an algorithmic reverse-engineering approach: studying existing research papers and reference implementations, understanding their underlying mathematics and computational architectures, and reimplementing them without relying on CUDA or high-level deep learning frameworks.

The pipeline is built around four major components:

- **SuperPoint** — Feature detection and description.
- **SuperGlue** — Feature matching.
- **Structure from Motion (SfM)** — Camera pose estimation and sparse reconstruction.
- **3D Gaussian Splatting** — 3D scene reconstruction and rendering.

## Tech Stack

- Visual Studio 2026
- C++17
- OpenGL 4.3+
- GLSL Compute Shaders

## Libraries and External Dependencies

- GLFW
- GLAD
- ImGui
- GLM
- stb_image

## Development Environment

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 5 5600X |
| GPU | AMD Radeon RX 6650 XT |
| RAM | 16 GB |

## Roadmap

- [x] ~~**SuperPoint — C++ / GLSL implementation**~~
- [x] ~~**SuperPoint — GPU optimization**~~
- [x] ~~**SuperGlue — C++ / GLSL implementation**~~
- [x] ~~**SuperGlue — C++ / GLSL implementation**~~
- [x] ~~**SuperPoint + SuperGlue integration**~~
- [ ] Feature matching and geometric verification (RANSAC)
- [ ] SfM — Camera pose estimation
- [ ] SfM — Feature triangulation
- [ ] Bundle Adjustment
- [ ] Sparse 3D reconstruction
- [ ] 3D Gaussian Splatting implementation
- [ ] Complete pipeline integration


## Reference Images

Refer to the [`ReferenceImages`](./ReferenceImages/) folder for explanatory diagrams and visual representations of the pipeline architecture and its individual components.


## References

- [SuperPoint — DeTone et al. (2018)](https://arxiv.org/abs/1712.07629)
- [SuperGlue — Sarlin et al. (2020)](https://arxiv.org/abs/1911.11763)

## Author

**Becem Mhamdi**

[GitHub](https://github.com/becem2)

---

<p align="center">
  <i>He who spares the wicked, harms the good .</i>
</p>