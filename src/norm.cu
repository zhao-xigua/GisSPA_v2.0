#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "fft.cuh"
#include "helper.cuh"
#include "kernels.cuh"
#include "norm.cuh"
#include "smartptr.cuh"

struct SearchNorm::impl {
  struct Device {
    struct FFT {
      cufftHandle templates;
      cufftHandle image;
      cufftHandle raw_image;
    } fft;

    cudaStream_t stream;

    device_unique_ptr<cufftComplex[]> padded_templates;
    device_unique_ptr<cufftComplex[]> CCG;
    device_unique_ptr<cufftComplex[]> CCG_sum;
    device_unique_ptr<cufftComplex[]> CCG_buf;

    device_unique_ptr<cufftComplex[]> padded_image;

    device_unique_ptr<float[]> image;
    device_unique_ptr<float[]> ra;
    device_unique_ptr<float[]> rb;
    device_unique_ptr<float[]> reduction_buf;
    device_unique_ptr<float[]> means;

    int id;
  } dev;

  struct Host {
    pinned_unique_ptr<float[]> reduction_buf;
    std::unique_ptr<float[]> ubuf;
  } host;
};

SearchNorm::SearchNorm(const Config& c, const EulerData& e, Size img, int device)
    : para(c),
      euler(e),
      padding_size(c.geti("window_size")),
      overlap(c.geti("overlap")),
      invert(c.geti("invert")),
      phase_flip(c.geti("phase_flip")),
      phi_step(c.getf("phistep")),
      batch_size(e.size()),
      nx(img.width),
      ny(img.height),
      line_count(0),
      pimpl(std::make_unique<impl>()) {
  std::cout << "Thread " << std::this_thread::get_id() << " set device " << device << "."
            << std::endl;
  pimpl->dev.id = device;
  cudaSetDevice(device);

  padded_template_size = padding_size * padding_size;

  auto blocks_one_axis = [](int length, int padding, int overlap) {
    std::vector<int> block_offsets{
        0,
    };
    int offset = 0;
    while (offset + padding < length) {
      offset += padding - overlap;
      if (offset + padding >= length) {
        offset = length - padding;
      }
      block_offsets.emplace_back(offset);
    }
    return block_offsets;
  };

  // num of blocks in x, y axis
  block_offsets_x = blocks_one_axis(nx, padding_size, overlap);
  block_offsets_y = blocks_one_axis(ny, padding_size, overlap);
  block_x = block_offsets_x.size();
  block_y = block_offsets_y.size();
  std::printf("Split block_x: %d, block_y: %d\n", block_x, block_y);

  // N = max{num of tmp, num of subimgs}
  batch_size = std::max(batch_size, static_cast<size_t>(block_x * block_y));

  pimpl->dev.padded_templates =
      make_device_unique<cufftComplex[]>(padded_template_size * batch_size);
  pimpl->dev.CCG = make_device_unique<cufftComplex[]>(padded_template_size * batch_size);
  pimpl->dev.CCG_sum = make_device_unique<cufftComplex[]>(padded_template_size * block_x * block_y);
  pimpl->dev.CCG_buf = make_device_unique<cufftComplex[]>(padded_template_size * batch_size);
  pimpl->dev.ra = make_device_unique<float[]>(batch_size * RA_SIZE);
  pimpl->dev.rb = make_device_unique<float[]>(batch_size * RA_SIZE);
  auto buf_size = 4 * padded_template_size * batch_size / BLOCK_SIZE;
  pimpl->host.reduction_buf = make_host_unique_pinned<float[]>(buf_size);
  pimpl->dev.reduction_buf = make_device_unique<float[]>(buf_size);
  pimpl->dev.means = make_device_unique<float[]>(batch_size);
  pimpl->host.ubuf = std::make_unique<float[]>(4 * batch_size);

  pimpl->dev.fft.templates = MakeFFTPlan(padding_size, padding_size, batch_size);
  cufftSetStream(pimpl->dev.fft.templates, pimpl->dev.stream);

  pimpl->dev.image = make_device_unique<float[]>(nx * ny);
  pimpl->dev.padded_image =
      make_device_unique<cufftComplex[]>(block_x * block_y * padded_template_size);

  pimpl->dev.fft.image = MakeFFTPlan(padding_size, padding_size, block_x * block_y);
  cufftSetStream(pimpl->dev.fft.image, pimpl->dev.stream);
  pimpl->dev.fft.raw_image = MakeFFTPlan(ny, nx, 1);
  cufftSetStream(pimpl->dev.fft.raw_image, pimpl->dev.stream);

  cudaStreamSynchronize(pimpl->dev.stream);

  cudaStreamCreate(&pimpl->dev.stream);
  CHECK();
  DeviceMemoryUsage();
}

SearchNorm::~SearchNorm() {
  cufftDestroy(pimpl->dev.fft.raw_image);
  cufftDestroy(pimpl->dev.fft.image);
  cufftDestroy(pimpl->dev.fft.templates);
  cudaStreamDestroy(pimpl->dev.stream);
}

void SearchNorm::LoadTemplate(const Templates& temp) {
  auto padded_templates = std::make_unique<cufftComplex[]>(padded_template_size * batch_size);
  std::memset(padded_templates.get(), 0, sizeof(cufftComplex) * padded_template_size * batch_size);

  // padding
  int sx = (padding_size - temp.width) / 2;
  int sy = (padding_size - temp.height) / 2;
  const size_t size = temp.width * temp.height;
#pragma omp parallel for
  for (int n = 0; n < batch_size; ++n) {
    for (int j = 0; j < temp.height; j++) {
      for (int i = 0; i < temp.width; i++) {
        size_t index = padded_template_size * n + (sy + j) * padding_size + (sx + i);
        float cur = temp.data[n * size + i + j * temp.width];
        padded_templates[index].x = cur;
      }
    }
  }
  cudaMemcpyAsync(pimpl->dev.padded_templates.get(), padded_templates.get(),
                  sizeof(cufftComplex) * padded_template_size * batch_size, cudaMemcpyHostToDevice,
                  pimpl->dev.stream);
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::LoadImage(const Image& img) {
  float sum = 0, sum_s2 = 0;
  for (int i = 0; i < nx * ny; i++) {
    float cur = img.data[i];
    sum += cur / nx / ny;
    sum_s2 += (cur * cur / nx / ny);
  }

  float avg = sum;
  float var = sqrt(sum_s2 - avg * avg);
  int up_bound = avg + 6 * var;
  int low_bound = avg - 6 * var;
  if (sum_s2 > avg * avg)
#pragma omp parallel for
    for (int i = 0; i < nx * ny; i++)
      if (img.data[i] > up_bound || img.data[i] < low_bound) img.data[i] = avg;

  if (invert) {
#pragma omp parallel for
    for (int i = 0; i < nx * ny; i++) img.data[i] = -img.data[i];
  }

  cudaMemcpyAsync(pimpl->dev.image.get(), img.data.get(), sizeof(float) * nx * ny,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::SetParams(const Image::Params& params) {
  // set params
  para.defocus = -params.defocus;
  para.dfang = params.dfang;
  para.dfdiff = params.dfdiff;
  para.dfu = -params.defocus + params.dfdiff;  // -defocus is minus, so abs(dfu) < abs(dfv)
  para.dfv = -params.defocus - params.dfdiff;
  para.lambda = 12.2639 / sqrt(para.energy * 1000.0 + 0.97845 * para.energy * para.energy);
  para.ds = 1 / (para.apix * padding_size);
}

void SearchNorm::PreprocessTemplate() {
  const int l = padding_size;
  const int nblocks = padded_template_size * batch_size / BLOCK_SIZE;

  float r = l / 2.f - 2.f;
  float up_bound{}, low_bound{};
  if (r > 1) {
    up_bound = (r + 1) * (r + 1);
    low_bound = (r - 1) * (r - 1);
  }

  cudaMemsetAsync(pimpl->dev.CCG.get(), 0, sizeof(cufftComplex) * padded_template_size * batch_size,
                  pimpl->dev.stream);
  cudaMemsetAsync(pimpl->dev.reduction_buf.get(), 0,
                  4 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  pimpl->dev.stream);

  // generate Mask, count number of non-zero digits
  generate_mask<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(
      l, pimpl->dev.CCG.get(), r, pimpl->dev.reduction_buf.get(), up_bound, low_bound);

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, batch_size * sizeof(float));

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    pimpl->host.ubuf[id] += pimpl->host.reduction_buf[k];
  }

  // Calculate dot of mask and all templates
  cudaMemcpyAsync(pimpl->dev.means.get(), pimpl->host.ubuf.get(), sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);

  multiCount_dot<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(
      l, pimpl->dev.CCG.get(), pimpl->dev.padded_templates.get(), pimpl->dev.means.get(),
      pimpl->dev.reduction_buf.get());

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, batch_size * sizeof(float));

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    pimpl->host.ubuf[id] += pimpl->host.reduction_buf[k];
  }

  UpdateSigma<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) * 2, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), pimpl->dev.reduction_buf.get());
  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  2 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  // put em on GPU
  cudaMemcpyAsync(pimpl->dev.means.get(), pimpl->host.ubuf.get(), sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    pimpl->host.ubuf[2 * id] += pimpl->host.reduction_buf[2 * k];
    // sum of value
    pimpl->host.ubuf[2 * id + 1] += pimpl->host.reduction_buf[2 * k + 1];
    // sum of value^2
  }

  auto sigmas = make_host_unique_pinned<double[]>(batch_size);
  for (int i = 0; i < batch_size; i++) {
    double mean = pimpl->host.ubuf[2 * i] / (double)(l * l);
    sigmas[i] = std::sqrt(pimpl->host.ubuf[2 * i + 1] / (double)(l * l) - mean * mean);
    if (sigmas[i] < 0 || !std::isfinite(sigmas[i])) sigmas[i] = 0;
  }

  auto dev_sigmas = make_device_unique<double[]>(batch_size);
  // data[i]=(data[i]-em)/s;
  cudaMemcpyAsync(dev_sigmas.get(), sigmas.get(), sizeof(double) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  scale_each<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      l, pimpl->dev.padded_templates.get(), pimpl->dev.means.get(), dev_sigmas.get());

  // **************************************************************
  // apply whitening filter and do ift
  // input: Padded IMAGE (Real SPACE)
  // output: IMAGE_whiten (Fourier SPACE in RI)
  // **************************************************************
  cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
  cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);

  // Inplace FFT
  cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(),
               pimpl->dev.padded_templates.get(), CUFFT_FORWARD);

  // CUFFT will enlarge VALUE to N times. Restore it
  scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padded_template_size * batch_size, padded_template_size);

  // Whiten at fourier space
  // contain ri2ap
  SQRSum_by_circle<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size,
      padding_size);

  // contain ap2ri
  whiten_Tmp<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size);

  // **************************************************************
  // 1. lowpass
  // 2. apply weighting function
  // 3. normlize
  // input: masked_whiten_IMAGE (Fourier SPACE in RI)
  // output: PROCESSED_IMAGE (Fourier SPACE in AP)
  // **************************************************************
  // contain ri2ap
  apply_weighting_function<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padding_size, para);

  compute_area_sum_ofSQR<<<nblocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float),
                           pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(),
                                                pimpl->dev.reduction_buf.get(), padding_size,
                                                padding_size);

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  2 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  // After Reduction -> compute mean for each image
  std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));
  float* infile_mean = pimpl->host.ubuf.get();
  float* counts = pimpl->host.ubuf.get() + batch_size;

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    infile_mean[id] += pimpl->host.reduction_buf[2 * k];
    counts[id] += pimpl->host.reduction_buf[2 * k + 1];
  }

  for (int k = 0; k < batch_size; k++) {
    infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
  }
  // Do Normalization with computed infile_mean[]
  cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  // Contain ap2ri
  normalize<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padding_size, padding_size, pimpl->dev.means.get());
  cudaMemsetAsync(pimpl->dev.means.get(), 0, batch_size * sizeof(float), pimpl->dev.stream);
  cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(),
               pimpl->dev.padded_templates.get(), CUFFT_INVERSE);

  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::PreprocessImage(const Image& img) {
  int nblocks = std::ceil(nx * ny / static_cast<double>(BLOCK_SIZE));

  cudaMemsetAsync(pimpl->dev.padded_image.get(), 0,
                  block_x * block_y * padded_template_size * sizeof(cufftComplex),
                  pimpl->dev.stream);

  float2Complex<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                               pimpl->dev.image.get(), nx, ny);
  // fft inplace
  cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_image.get(),
               pimpl->dev.padded_image.get(), CUFFT_FORWARD);
  scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), nx * ny,
                                                       nx * ny);

  // phase flip
  do_phase_flip<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), para,
                                                               nx, ny);

  // Whiten at fourier space
  cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
  cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);

  // contain ri2ap
  SQRSum_by_circle<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, 1);

  // 1. whiten
  // 2. low pass
  // 3. weight
  // 4. ap2ri
  whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, para);
  // 0Hz -> 0
  set_0Hz_to_0_at_RI<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                                    nx, ny);

  // ifft inplace
  cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_image.get(),
               pimpl->dev.padded_image.get(), CUFFT_INVERSE);
  Complex2float<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.image.get(), pimpl->dev.padded_image.get(), nx, ny);

  cudaStreamSynchronize(pimpl->dev.stream);
  pimpl->dev.ra = nullptr;
  pimpl->dev.rb = nullptr;
}

void SearchNorm::SplitImage() {
  int l = padding_size;
  // Init d_rotated_imge to all {0}
  int ix = block_x;
  int iy = block_y;
  int nblocks = ix * iy * padded_template_size / BLOCK_SIZE;

  cudaMemsetAsync(pimpl->dev.padded_image.get(), 0,
                  ix * iy * padded_template_size * sizeof(cufftComplex), pimpl->dev.stream);

  auto d_off_x = make_device_unique<int[]>(block_offsets_x.size());
  auto d_off_y = make_device_unique<int[]>(block_offsets_y.size());
  cudaMemcpyAsync(d_off_x.get(), block_offsets_x.data(), sizeof(int) * block_offsets_x.size(),
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  cudaMemcpyAsync(d_off_y.get(), block_offsets_y.data(), sizeof(int) * block_offsets_y.size(),
                  cudaMemcpyHostToDevice, pimpl->dev.stream);

  // split Image into blocks with overlap
  split_IMG<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.image.get(), pimpl->dev.padded_image.get(), d_off_x.get(), d_off_y.get(), nx, ny,
      padding_size, block_x, overlap);

  cudaStreamSynchronize(pimpl->dev.stream);
  pimpl->dev.image = nullptr;

  // do normalize to all subIMGs
  // Inplace FFT
  cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(),
               CUFFT_FORWARD);
  // Scale IMG to normal size
  scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                       ix * iy * padded_template_size, l * l);
  ri2ap<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                       ix * iy * padded_template_size);
  compute_area_sum_ofSQR<<<nblocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float),
                           pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                pimpl->dev.reduction_buf.get(), l, l);

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  4 * padded_template_size * batch_size / BLOCK_SIZE * sizeof(float),
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  int nimg = block_x * block_y;
  // After Reduction -> compute mean for each image
  float infile_mean[nimg], counts[nimg];
  std::memset(infile_mean, 0, sizeof(float) * nimg);
  std::memset(counts, 0, sizeof(float) * nimg);
  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < nblocks; k++) {
    int id = k / ((l * l) / BLOCK_SIZE);
    infile_mean[id] += pimpl->host.reduction_buf[2 * k];
    counts[id] += pimpl->host.reduction_buf[2 * k + 1];
  }
  for (int k = 0; k < nimg; k++) {
    infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
  }

  // Do Normalization with computed infile_mean[]
  cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * nimg, cudaMemcpyHostToDevice,
                  pimpl->dev.stream);
  // Contain ap2ri
  normalize<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), l, l,
                                                           pimpl->dev.means.get());

  // Inplace IFT
  // cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_image, pimpl->dev.padded_image,
  // CUFFT_INVERSE);

  cudaMemsetAsync(pimpl->dev.CCG_sum.get(), 0,
                  sizeof(cufftComplex) * padded_template_size * block_x * block_y,
                  pimpl->dev.stream);

  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::RotateTemplate(float euler3) {
  int l = padding_size;
  int nblocks = padded_template_size * batch_size / BLOCK_SIZE;

  rotate_subIMG<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(),
                                                               pimpl->dev.CCG_buf.get(), euler3, l);
  apply_mask<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), para.d_m,
                                                            para.edge_half_width, padding_size);
  compute_sum_sqr<<<nblocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(
      pimpl->dev.CCG_buf.get(), pimpl->dev.reduction_buf.get(), padding_size, padding_size);
  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  2 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));
  float* infile_mean = pimpl->host.ubuf.get();
  float* infile_sqr = pimpl->host.ubuf.get() + batch_size;

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    infile_mean[id] += pimpl->host.reduction_buf[2 * k];
    infile_sqr[id] += pimpl->host.reduction_buf[2 * k + 1];
  }

  for (int k = 0; k < batch_size; k++) {
    infile_mean[k] = (infile_mean[k] / (padded_template_size));
    infile_sqr[k] = infile_sqr[k] / padded_template_size - infile_mean[k] * infile_mean[k];
  }
  cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  substract_by_mean<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), l, l,
                                                                   pimpl->dev.means.get());
  cudaMemcpyAsync(pimpl->dev.means.get(), infile_sqr, sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  divided_by_var<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.CCG_buf.get(), padding_size, padding_size, pimpl->dev.means.get());
  cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG_buf.get(), pimpl->dev.CCG_buf.get(),
               CUFFT_FORWARD);
  scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.CCG_buf.get(), padded_template_size * batch_size, padded_template_size);
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::ComputeCCGSum() {
  int l = padding_size;
  int nblocks = padded_template_size * batch_size / BLOCK_SIZE;

  // compute score for each block
  for (int j = 0; j < block_y; ++j) {
    for (int i = 0; i < block_x; ++i) {
      auto block_id = i + j * block_x;
      // compute CCG
      compute_corner_CCG<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
          pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.padded_image.get(), l,
          block_id);
      // Inplace IFT
      cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(),
                   CUFFT_INVERSE);
      // compute avg/variance
      add_CCG_to_sum<<<padded_template_size / BLOCK_SIZE, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
          pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), l, batch_size, block_id);
    }
  }
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::ComputeCCGMean() {
  int nblocks = padded_template_size * block_x * block_y / BLOCK_SIZE;

  set_CCG_mean<<<nblocks, BLOCK_SIZE>>>(pimpl->dev.CCG_sum.get(), padding_size, batch_size,
                                        360 / phi_step);
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::PickParticles(std::vector<float>& scores, float euler3) {
  int l = padding_size;
  int nblocks = padded_template_size * batch_size / BLOCK_SIZE;

  scores.clear();

  // compute score for each block
  for (int j = 0; j < block_y; j++) {
    for (int i = 0; i < block_x; i++) {
      // compute CCG
      compute_corner_CCG<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
          pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.padded_image.get(), l,
          i + j * block_x);
      // Inplace IFT
      cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(),
                   CUFFT_INVERSE);
      // update CCG with avg/var
      update_CCG<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
          pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), l, i + j * block_x);

      // find peak in each block
      get_peak_pos<<<nblocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(
          pimpl->dev.CCG.get(), pimpl->dev.reduction_buf.get(), l);
      cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                      2 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                      cudaMemcpyDeviceToHost, pimpl->dev.stream);
      cudaStreamSynchronize(pimpl->dev.stream);

      // After Reduction -> compute mean for each image
      for (int k = 0; k < (padded_template_size * batch_size) / BLOCK_SIZE; k++) {
        int J = k / (padded_template_size / BLOCK_SIZE);
        if (pimpl->host.reduction_buf[2 * k] >= para.thresh) {
          float score = pimpl->host.reduction_buf[2 * k];
          int centerx = block_offsets_x[i] + (int)pimpl->host.reduction_buf[2 * k + 1] % l;
          int centery = block_offsets_y[j] + (int)pimpl->host.reduction_buf[2 * k + 1] / l;
          if (centerx >= para.d_m && centerx < nx - para.d_m && centery >= para.d_m &&
              centery < ny - para.d_m) {
            scores.emplace_back(score);
            scores.emplace_back(centerx);
            scores.emplace_back(centery);
            scores.emplace_back(J);
          }
        }
      }
    }
  }
}

void SearchNorm::OutputScore(std::ostream& output, std::vector<float>& scores, float euler3,
                             const Image& input) {
  char buf[1024];
  for (int i = 0; i < scores.size(); i += 4) {
    float score = scores[i];
    float centerx = scores[i + 1];
    float centery = scores[i + 2];
    size_t j = scores[i + 3];
    std::snprintf(
        buf, 1024,
        "%d\t%s\tdefocus=%f\tdfdiff=%f\tdfang=%f\teuler=%f,%f,%f\tcenter=%f,%f\tscore=%f\n",
        input.unused, input.rpath.c_str(), -para.defocus, para.dfdiff, para.dfang, euler.euler1[j],
        euler.euler2[j], euler3, centerx, centery, score);
    output << buf;
    ++line_count;
  }
}

void SearchNorm::work(const Templates& temp, const Image& image, std::ostream& output) {
  std::vector<float> scores;
  auto params = image.p;

  SetParams(params);
  std::printf("Device %d: Load template\n", pimpl->dev.id);
  LoadTemplate(temp);

  std::printf("Device %d: Image: %s, (%zu, %zu)\n", pimpl->dev.id, image.rpath.c_str(),
              params.width, params.height);
  std::printf("Device %d: Load image\n", pimpl->dev.id);
  LoadImage(image);

  std::printf("Device %d: Preprocess template\n", pimpl->dev.id);
  PreprocessTemplate();
  std::printf("Device %d: Preprocess image\n", pimpl->dev.id);
  PreprocessImage(image);
  std::printf("Device %d: Split image\n", pimpl->dev.id);
  SplitImage();

  std::printf("Device %d: Compute the avgs and vars of all CCGs, euler3 in [0, 360), step = %f\n",
              pimpl->dev.id, phi_step);
  for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) {
    RotateTemplate(euler3);
    ComputeCCGSum();
  }
  std::printf("\n");
  std::printf("Device %d: Compute CCG means\n", pimpl->dev.id);
  ComputeCCGMean();
  std::printf("Device %d: Update CCGs and compute scores, euler3 in [0, 360), step = %f\n",
              pimpl->dev.id, phi_step);
  for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) {
    RotateTemplate(euler3);
    PickParticles(scores, euler3);
    float cur_e = 360.0f - euler3;
    if (cur_e >= 360.0f) cur_e -= 360.0f;
    OutputScore(output, scores, cur_e, image);
  };
  std::printf("\n");
  std::printf("Device %d: Current output line count: %d\n", pimpl->dev.id, line_count);
}

void SearchNorm::work_verbose(const Templates& temp, const Image& image, std::ostream& output) {
  std::vector<float> scores;
  auto params = image.p;

  SetParams(params);
  std::printf("Device %d: Load template\n", pimpl->dev.id);
  LoadTemplate(temp);

  std::printf("Device %d: Image: %s, (%zu, %zu)\n", pimpl->dev.id, image.rpath.c_str(),
              params.width, params.height);
  std::printf("Device %d: Load image\n", pimpl->dev.id);
  LoadImage(image);

  std::printf("Device %d: Preprocess template\n", pimpl->dev.id);
  PreprocessTemplate();
  std::printf("Device %d: Preprocess image\n", pimpl->dev.id);
  PreprocessImage(image);
  std::printf("Device %d: Split image\n", pimpl->dev.id);
  SplitImage();

  std::printf("Device %d: Compute the avgs and vars of all CCGs, euler3 in [0, 360), step = %f\n",
              pimpl->dev.id, phi_step);
  for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) {
    RotateTemplate(euler3);
    ComputeCCGSum();
    if (static_cast<int>(euler3) % 9 == 0) {
      std::printf(".");
      std::fflush(stdout);
    }
  }
  std::printf("\n");
  std::printf("Device %d: Compute CCG means\n", pimpl->dev.id);
  ComputeCCGMean();
  std::printf("Device %d: Update CCGs and compute scores, euler3 in [0, 360), step = %f\n",
              pimpl->dev.id, phi_step);
  for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) {
    RotateTemplate(euler3);
    PickParticles(scores, euler3);
    float cur_e = 360.0f - euler3;
    if (cur_e >= 360.0f) cur_e -= 360.0f;
    OutputScore(output, scores, cur_e, image);
    if (static_cast<int>(euler3) % 9 == 0) {
      std::printf(".");
      std::fflush(stdout);
    }
  };
  std::printf("\n");
  std::printf("Device %d: Current output line count: %d\n", pimpl->dev.id, line_count);
}