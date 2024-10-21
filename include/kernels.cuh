#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include "constants.h"
#include "DataReader2.h"

__global__ void UpdateSigma(cufftComplex* d_templates, float* d_buf);
__global__ void generate_mask(int l, cufftComplex* mask, float r, float* d_buf, float up, float low);
__global__ void multiCount_dot(int l, cufftComplex* mask, cufftComplex* d_templates, float* constants,
                               float* res);
__global__ void scale_each(int l, cufftComplex* d_templates, float* ems, double* d_sigmas);
__global__ void SQRSum_by_circle(cufftComplex* data, float* ra, float* rb, int nx, int ny, int mode = 0);
__global__ void whiten_Tmp(cufftComplex* data, float* ra, float* rb, int l);
__global__ void whiten_filter_weight_Img(cufftComplex* data, float* ra, float* rb, int nx, int ny, Parameters para);
__global__ void set_0Hz_to_0_at_RI(cufftComplex* data, int nx, int ny);
__global__ void apply_mask(cufftComplex* data, float d_m, float edge_half_width, int l);
__global__ void apply_weighting_function(cufftComplex* data, size_t padding_size, Parameters para);
__global__ void compute_area_sum_ofSQR(cufftComplex* data, float* res, int nx, int ny);
__global__ void compute_sum_sqr(cufftComplex* data, float* res, int nx, int ny);
__global__ void normalize(cufftComplex* d_templates, int nx, int ny, float* means);
__global__ void divided_by_var(cufftComplex* d_templates, int nx, int ny, float* var);
__global__ void substract_by_mean(cufftComplex* d_templates, int nx, int ny, float* means);
__global__ void rotate_IMG(float* d_image, float* d_rotated_image, float e, int nx, int ny);
__global__ void rotate_subIMG(cufftComplex* d_image, cufftComplex* d_rotated_image, float e, int l);
__global__ void split_IMG(float* Ori, cufftComplex* IMG, int nx, int ny, int l, int bx, int overlap);
__global__ void split_IMG(float* Ori, cufftComplex* IMG, int* block_off_x, int* block_off_y, int nx, int ny, int l,
                          int bx, int overlap);
__global__ void compute_corner_CCG(cufftComplex* CCG, cufftComplex* Tl, cufftComplex* IMG, int l,
                                   int block_id);
__global__ void add_CCG_to_sum(cufftComplex* CCG_sum, cufftComplex* CCG, int l, int N_tmp, int block_id);
__global__ void set_CCG_mean(cufftComplex* CCG_sum, int l, int N_tmp, int N_euler);
__global__ void update_CCG(cufftComplex* CCG_sum, cufftComplex* CCG, int l, int block_id);
__global__ void get_peak_and_SUM(cufftComplex* odata, float* res, int l, float d_m);
__global__ void get_peak_pos(cufftComplex* odata, float* res, int l);
__global__ void scale(cufftComplex* data, size_t size, int l2);
__global__ void ri2ap(cufftComplex* data, size_t size);
__global__ void ap2ri(cufftComplex* data);
__global__ void Complex2float(float* f, cufftComplex* c, int nx, int ny);
__global__ void float2Complex(cufftComplex* c, float* f, int nx, int ny);
__global__ void do_phase_flip(cufftComplex* filter, Parameters para, int nx, int ny);
__device__ float CTF_AST(int x1, int y1, int nx, int ny, float apix, float dfu, float dfv, float dfdiff, float dfang,
                         float lambda, float cs, float ampconst, int mode);
