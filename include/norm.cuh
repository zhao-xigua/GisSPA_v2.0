#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "DataReader2.h"
#include "emdata.h"
#include "image.hpp"
#include "templates.hpp"

struct SearchNorm {
  Parameters para;
  EulerData euler;

  struct Size {
    size_t width;
    size_t height;
  };

  struct impl;
  std::unique_ptr<impl> pimpl;

  std::vector<int> block_offsets_x, block_offsets_y;

  size_t padded_template_size;
  size_t batch_size;
  int padding_size;
  int overlap;
  int nx, ny;
  int block_x, block_y;
  int line_count;
  float phi_step;
  bool invert, phase_flip;
  bool image_dependent_allocated;

  SearchNorm(const Config& c, const EulerData& e, Size img, int device = 0);
  ~SearchNorm();

  void LoadTemplate(const Templates& temp);
  void LoadImage(const Image& img);
  void SetParams(const Image::Params& params);
  void PreprocessTemplate();
  void PreprocessImage(const Image& img);
  void SplitImage();
  void RotateTemplate(float euler3);
  void ComputeCCGSum();
  void ComputeCCGMean();
  void PickParticles(std::vector<float>& scores, float euler3);
  void OutputScore(std::ostream& output, std::vector<float>& scores, float euler3,
                   const Image& input);

  void work(const Templates& temp, const Image& image, std::ostream& output);
  void work_verbose(const Templates& temp, const Image& image, std::ostream& output);
};