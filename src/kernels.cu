#include "constants.h"
#include "kernels.cuh"

__global__ void UpdateSigma(cufftComplex* d_templates, float* d_buf) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  int tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = d_templates[i].x;
  sdata[tid + blockDim.x] = d_templates[i].x * d_templates[i].x;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      // sum of data[i] & data[i]^2
      sdata[tid] += sdata[tid + s];
      sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_buf[blockIdx.x * 2] = sdata[0];
    d_buf[blockIdx.x * 2 + 1] = sdata[blockDim.x];
  }
}

__global__ void generate_mask(int l, cufftComplex* mask, float r, float* res, float up, float low) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  int tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;

  // Dis^2 between (x,y) and center (l/2,l/2)
  float rr = (x - l / 2) * (x - l / 2) + (y - l / 2) * (y - l / 2);
  if (rr >= low && rr <= up) {
    mask[i].x = 1;
    mask[i].y = 0;
  }

  // reduction for number of non-zero digits
  sdata[tid] = mask[i].x;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) res[blockIdx.x] = sdata[0];
}

__global__ void multiCount_dot(int l, cufftComplex* mask, cufftComplex* d_templates,
                               float* constants, float* res) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  int tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int image_id = i / image_size;

  // Multi constant 1/non-zeros
  if (constants[image_id] != 0) mask[i].x *= 1.0 / constants[image_id];

  // reduction for dot
  sdata[tid] = mask[i].x * d_templates[i].x;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // use res to store dot
  if (tid == 0) res[blockIdx.x] = sdata[0];
}

__global__ void scale_each(int l, cufftComplex* d_templates, float* ems, double* d_sigmas) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int image_id = i / image_size;

  if (d_sigmas[image_id] - 0 < EPS && d_sigmas[image_id] - 0 > -EPS) return;
  d_templates[i].x = (d_templates[i].x - ems[image_id]) / d_sigmas[image_id];
}

// mode = 0 (default)  for template
// mode = 1 for raw image
__global__ void SQRSum_by_circle(cufftComplex* data, float* ra, float* rb, int nx, int ny,
                                 int mode) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = nx * ny;
  if (mode == 1 && i >= image_size) return;

  int local_id = i % image_size;
  int x = local_id % nx;
  int y = local_id / nx;

  float tmp;
  // ri2ap
  tmp = hypotf(data[i].x, data[i].y);
  if (data[i].x == 0 && data[i].y == 0)
    data[i].y = 0;
  else
    data[i].y = atan2(data[i].y, data[i].x);
  data[i].x = tmp;

  if (x > nx / 2) return;

  // calculate the number of point with fixed distance ('r') from center
  int r = floor(hypotf(min(y, ny - y), min(x, nx - x)) + 0.5) - 1;

  if (r < max(nx, ny) / 2 && r >= 0) {
    // Add offset
    r += RA_SIZE * (i / image_size);
    atomicAdd(&ra[r], data[i].x * data[i].x);
    atomicAdd(&rb[r], 1.0);
  }
}

__global__ void whiten_Tmp(cufftComplex* data, float* ra, float* rb, int l) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;
  int r = floor(hypotf(min(y, l - y), min(x, l - x)) + 0.5) - 1;

  if (r < l / 2 && r >= 0) {
    // Add offset
    r += RA_SIZE * (i / image_size);
    float fb_infile = ra[r] / rb[r];
    data[i].x = data[i].x / (float)sqrt(fb_infile);
  }

  // ap2ri
  float tmp = data[i].x * sinf(data[i].y);
  data[i].x = data[i].x * cosf(data[i].y);
  data[i].y = tmp;
}

__global__ void whiten_filter_weight_Img(cufftComplex* data, float* ra, float* rb, int nx, int ny,
                                         Parameters para) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = nx * ny;
  if (i >= image_size) return;

  int local_id = i % image_size;
  int x = local_id % nx;
  int y = local_id / nx;

  float dx = min(x, nx - x);
  float dy = min(y, ny - y);
  int r = floor(hypotf(dx, dy) + 0.5) - 1;
  float ss =
      sqrtf((dx * dx / (float)(nx * nx) + dy * dy / (float)(ny * ny)) / (para.apix * para.apix));
  int l = max(nx, ny);
  float v, signal, Ncurve;

  // whiten
  if (r < l / 2 && r >= 0) {
    v = CTF_AST(x, (y + ny / 2) % ny, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff,
                para.dfang, para.lambda, para.cs, para.ampconst, 2);
    signal = exp(para.bfactor * ss * ss + para.bfactor2 * ss + para.bfactor3);
    Ncurve = exp(para.a * ss * ss + para.b * ss + para.b2);

    data[i].x = data[i].x * sqrt((signal * v * v + Ncurve) / signal) / sqrt(ra[r] / rb[r]);
    if (r > (l * para.apix / 6)) data[i].x = data[i].x * exp(-100 * ss * ss);
  }

  // low pass
  if (r < l * para.apix / para.highres && r >= l * para.apix / para.lowres) {
  } else if (r >= l * para.apix / para.highres && r < l * para.apix / para.highres + 8) {
    data[i].x = data[i].x * (0.5 * cosf(PI * (r - l * para.apix / para.highres) / (2 * 8)) + 0.5);
  } else if (r >= (l * para.apix / para.lowres - 8) && r < l * para.apix / para.lowres && r >= 0) {
    data[i].x = data[i].x * (0.5 * cosf(PI * (l * para.apix / para.lowres - r) / (2 * 8)) + 0.5);
  } else
    data[i].x = 0;

  // apply weighting function
  if (r < l / 2 && r >= 0) {
    signal /= (para.kk + 1);
    Ncurve /= signal;
    data[i].x = data[i].x * sqrt(1 / (Ncurve + para.kk * v * v));
  }

  // ap2ri
  float tmp = data[i].x * sinf(data[i].y);
  data[i].x = data[i].x * cosf(data[i].y);
  data[i].y = tmp;
}

__global__ void set_0Hz_to_0_at_RI(cufftComplex* data, int nx, int ny) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 1) return;

  // ri2ap
  float tmp = hypotf(data[i].x, data[i].y);
  if (data[i].x == 0 && data[i].y == 0)
    data[i].y = 0;
  else
    data[i].y = atan2(data[i].y, data[i].x);
  data[i].x = tmp;

  if (i == 0) data[i].x = 0;

  // ap2ri
  tmp = data[i].x * sinf(data[i].y);
  data[i].x = data[i].x * cosf(data[i].y);
  data[i].y = tmp;
}

__global__ void apply_mask(cufftComplex* data, float d_m, float edge_half_width, int l) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;
  d_m = 1.5 * d_m;
  float r = hypotf(x - l / 2, y - l / 2);
  if (r > (d_m / 2 + 2 * edge_half_width)) {
    data[i].x = 0;
  } else if (r >= d_m / 2) {
    float d = 0.5 * cosf(PI * (r - d_m / 2) / (2 * edge_half_width)) + 0.5;
    data[i].x *= d;
  }
}

__global__ void apply_weighting_function(cufftComplex* data, size_t padding_size, Parameters para) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int l = padding_size;
  int image_size = l * l;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;

  float tmp;
  // ri2ap
  tmp = hypotf(data[i].x, data[i].y);
  if (data[i].x == 0 && data[i].y == 0)
    data[i].y = 0;
  else
    data[i].y = atan2(data[i].y, data[i].x);
  data[i].x = tmp;

  // low pass
  float r = hypotf(min(y, l - y), min(x, l - x));
  int r_round = floor(r + 0.5) - 1;
  if (r_round < l * para.apix / para.highres && r_round >= 0) {
  } else if (r_round >= l * para.apix / para.highres &&
             r_round < l * para.apix / para.highres + 8) {
    data[i].x =
        data[i].x * (0.5 * cosf(PI * (r_round - l * para.apix / para.highres) / (2 * 8)) + 0.5);
  } else if (r_round >= (l * para.apix / para.lowres - 8) &&
             r_round < l * para.apix / para.lowres && r_round >= 0) {
    data[i].x =
        data[i].x * (0.5 * cosf(PI * (l * para.apix / para.lowres - r_round) / (2 * 8)) + 0.5);
  } else
    data[i].x = 0;
  float ss = r * para.ds;

  float v, signal, Ncurve;
  // apply weighting function
  if (r_round < l / 2 && r_round >= 0) {
    v = CTF_AST(x, (y + l / 2) % l, l, l, para.apix, para.dfu, para.dfv, para.dfdiff, para.dfang,
                para.lambda, para.cs, para.ampconst, 2);
    signal = (exp(para.bfactor * ss * ss + para.bfactor2 * ss + para.bfactor3)) / (para.kk + 1);
    Ncurve = exp(para.a * ss * ss + para.b * ss + para.b2) / signal;
    // euler_w[x]=1.68118*ss;
    data[i].x = data[i].x * v * sqrt(1 / (Ncurve + para.kk * v * v));
  }
}

__device__ float CTF_AST(int x1, int y1, int nx, int ny, float apix, float dfu, float dfv,
                         float dfdiff, float dfang, float lambda, float cs, float ampconst,
                         int mode) {
  float v, ss, ag, gamma, df_ast;
  y1 = x1 > nx / 2 ? ny - y1 : y1;
  x1 = min(x1, nx - x1);
  float dx = min(x1, nx - x1);
  float dy = y1 - ny / 2;
  ss = (dx * dx / (float)(nx * nx) + dy * dy / (float)(ny * ny)) / (apix * apix);
  ag = atan2(float(y1 - ny / 2), float(x1));

  df_ast = 0.5 * (dfu + dfv + 2 * dfdiff * cosf(2 * (dfang * PI / 180 - ag)));
  gamma =
      -2 * PI * (cs * 2.5e6 * lambda * lambda * lambda * ss * ss + df_ast * 5000.0 * lambda * ss);
  if (mode == 0) {
    v = (sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma)) > 0
            ? 1.0
            : -1.0;  // do phaseflip
  } else if (mode == 2) {
    v = fabs(sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) +
             ampconst * cosf(gamma));  //	return abs ctf value
  } else {
    v = (sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) +
         ampconst * cosf(gamma));  //	return ctf value
  }

  return v;
}

__global__ void compute_area_sum_ofSQR(cufftComplex* data, float* res, int nx, int ny) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  auto tid = threadIdx.x;
  int image_size = nx * ny;
  int local_id = i % image_size;
  int x = local_id % nx;
  int y = local_id / nx;
  int r = floor(hypotf(min(y, ny - y), min(x, nx - x)) + 0.5) - 1;
  int l = max(nx, ny);

  if (r < l / 2 && r >= 0 && x <= nx / 2) {
    sdata[tid] = data[i].x * data[i].x;
    sdata[tid + blockDim.x] = 1;
  } else {
    sdata[tid] = 0;
    sdata[tid + blockDim.x] = 0;
  }
  __syncthreads();

  // if (tid < 512) {
  //   sdata[tid] += sdata[tid + 512];
  //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 512];
  // }
  // __syncthreads();
  // if (tid < 256) {
  //   sdata[tid] += sdata[tid + 256];
  //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 256];
  // }
  // __syncthreads();
  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 128];
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 64];
  }
  __syncthreads();

  if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 2];
    sdata[tid] += sdata[tid + 1];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 1];
  }

  // write result for this block
  if (tid == 0) {
    res[2 * blockIdx.x] = sdata[0];
    res[2 * blockIdx.x + 1] = sdata[blockDim.x];
  }
}

__global__ void compute_sum_sqr(cufftComplex* data, float* res, int nx, int ny) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  auto tid = threadIdx.x;
  int image_size = nx * ny;
  int local_id = i % image_size;
  int x = local_id % nx;
  int y = local_id / nx;
  int r = floor(hypotf(min(y, ny - y), min(x, nx - x)) + 0.5) - 1;
  int l = max(nx, ny);

  sdata[tid] = data[i].x;
  sdata[tid + blockDim.x] = data[i].x * data[i].x;

  __syncthreads();

  // if (tid < 512) {
  //   sdata[tid] += sdata[tid + 512];
  //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 512];
  // }
  // __syncthreads();
  // if (tid < 256) {
  //   sdata[tid] += sdata[tid + 256];
  //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 256];
  // }
  // __syncthreads();
  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 128];
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 64];
  }
  __syncthreads();

  if (tid < 32) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 2];
    sdata[tid] += sdata[tid + 1];
    sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 1];
  }

  // write result for this block
  if (tid == 0) {
    res[2 * blockIdx.x] = sdata[0];
    res[2 * blockIdx.x + 1] = sdata[blockDim.x];
  }
}

__global__ void normalize(cufftComplex* data, int nx, int ny, float* means) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = nx * ny;
  int template_id = i / image_size;

  if (means[template_id] != 0) data[i].x = data[i].x / means[template_id];

  // ap2ri
  float tmp = data[i].x * sinf(data[i].y);
  data[i].x = data[i].x * cosf(data[i].y);
  data[i].y = tmp;
}

__global__ void divided_by_var(cufftComplex* data, int nx, int ny, float* var) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  auto image_size = nx * ny;
  auto template_id = i / image_size;

  if (var[template_id] != 0) data[i].x = data[i].x / sqrtf(var[template_id]);
}
__global__ void substract_by_mean(cufftComplex* data, int nx, int ny, float* means) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  auto image_size = nx * ny;
  auto template_id = i / image_size;

  if (means[template_id] != 0) data[i].x = data[i].x - means[template_id];
}

__global__ void rotate_IMG(float* d_image, float* d_rotated_image, float e, int nx, int ny) {
  float cose = cos(e * PI / 180);
  float sine = sin(e * PI / 180);
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx, i = id % nx;
  float y = j - ny / 2, x = i - nx / 2;
  if (i >= nx || j >= ny) return;

  // Res of rotation from (x,y)
  float res = 0;

  //(x,y) rotate e with (nx/2,ny/2) (clockwise)
  float x2 = (cose * x + sine * y) + nx / 2;
  float y2 = (-sine * x + cose * y) + ny / 2;

  // Ouf of boundary after rotation
  if (x2 < 0 || x2 > nx - 1.0 || y2 < 0 || y2 > ny - 1.0)
    res = 0;
  else {
    int ii, jj;
    int k0, k1, k2, k3;
    float t, u, p0, p1, p2, p3;
    ii = floor(x2);
    jj = floor(y2);
    k0 = ii + jj * nx;
    k1 = k0 + 1;
    k2 = k0 + nx + 1;
    k3 = k0 + nx;

    // handle situation when ii,jj are out of boundary
    if (ii == nx - 1) {
      k1--;
      k2--;
    }
    if (jj == ny - 1) {
      k2 -= nx;
      k3 -= nx;
    }
    t = (x2 - (float)ii);
    u = (y2 - (float)jj);
    float tt = 1.0 - t;
    float uu = 1.0 - u;

    // bilinear interpolation of raw data (i,j)(i+1,j)(i,j+1)(i+1,j+1)
    p0 = d_image[k0] * tt * uu;
    p1 = d_image[k1] * t * uu;
    p3 = d_image[k3] * tt * u;
    p2 = d_image[k2] * t * u;
    res = p0 + p1 + p2 + p3;
  }

  // res <=> data[i+j*nx] after rotation
  d_rotated_image[id] = res;
}

__global__ void rotate_subIMG(cufftComplex* d_image, cufftComplex* d_rotated_image, float e,
                              int l) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;

  float cose = cos(e * PI / 180);
  float sine = sin(e * PI / 180);

  int image_size = l * l;
  int local_id = id % image_size;
  int off = id - local_id;

  int i = local_id % l;
  int j = local_id / l;
  int nx = l, ny = l;
  float y = j - ny / 2, x = i - nx / 2;

  // Res of rotation from (x,y)
  float res_x = 0, res_y = 0;

  //(x,y) rotate e with (nx/2,ny/2) (clockwise)
  float x2 = (cose * x + sine * y) + nx / 2;
  float y2 = (-sine * x + cose * y) + ny / 2;

  // Ouf of boundary after rotation
  if (x2 < 0 || x2 > nx - 1.0 || y2 < 0 || y2 > ny - 1.0) {
    res_x = 0;
    res_y = 0;
  } else {
    int ii, jj;
    int k0, k1, k2, k3;
    float t, u, p0, p1, p2, p3;
    ii = floor(x2);
    jj = floor(y2);
    k0 = ii + jj * nx;
    k1 = k0 + 1;
    k2 = k0 + nx + 1;
    k3 = k0 + nx;

    // handle situation when ii,jj are out of boundary
    if (ii == nx - 1) {
      k1 = k1 - 1;
      k2 = k2 - 1;
    }
    if (jj == ny - 1) {
      k2 -= nx;
      k3 -= nx;
    }
    t = x2 - ii;
    u = y2 - jj;
    float tt = 1.0 - t;
    float uu = 1.0 - u;

    // bilinear interpolation of raw data (i,j)(i+1,j)(i,j+1)(i+1,j+1)
    p0 = d_image[off + k0].x * tt * uu;
    p1 = d_image[off + k1].x * t * uu;
    p3 = d_image[off + k3].x * tt * u;
    p2 = d_image[off + k2].x * t * u;
    res_x = p0 + p1 + p2 + p3;

    p0 = d_image[off + k0].y * tt * uu;
    p1 = d_image[off + k1].y * t * uu;
    p3 = d_image[off + k3].y * tt * u;
    p2 = d_image[off + k2].y * t * u;
    res_y = p0 + p1 + p2 + p3;
  }

  // res <=> data[i+j*nx] after rotation
  d_rotated_image[id].x = res_x;
  d_rotated_image[id].y = res_y;
}

__global__ void split_IMG(float* Ori, cufftComplex* IMG, int nx, int ny, int l, int bx,
                          int overlap) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int image_id = i / image_size;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;

  int tmp = l - overlap;

  int area_x_id = image_id % bx;
  int area_y_id = image_id / bx;
  int ori_x = area_x_id * tmp + x;
  int ori_y = area_y_id * tmp + y;

  if (ori_x >= nx || ori_y >= ny) return;
  IMG[i].x = Ori[ori_x + ori_y * nx];
}

__global__ void split_IMG(float* Ori, cufftComplex* IMG, int* block_off_x,
                          int* block_off_y, int nx, int ny, int l, int bx, int overlap) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int image_size = l * l;
  int image_id = i / image_size;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;

  int area_x_id = image_id % bx;
  int area_y_id = image_id / bx;
  int ori_x = block_off_x[area_x_id] + x;
  int ori_y = block_off_y[area_y_id] + y;

  if (ori_x >= nx || ori_y >= ny) return;
  IMG[i].x = Ori[ori_x + ori_y * nx];
}

__global__ void compute_corner_CCG(cufftComplex* CCG, cufftComplex* Tl, cufftComplex* IMG, int l,
                                   int block_id) {
  // On this function,block means subimage splitted from IMG, not block ON GPU
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  // Area of rectangle, l^2
  int l2 = l * l;

  // Local id corresponding to splitted IMG
  int local_id = i % l2;
  int local_x = local_id % l;
  int local_y = local_id / l;

  int off = block_id * l2;

  // Global ID in IMG
  int j = local_x + local_y * l + off;

  // CCG[i] = IMG'[i]*template[i]
  //  ' means conjugate
  CCG[i].x = (IMG[j].x * Tl[i].x + IMG[j].y * Tl[i].y);
  CCG[i].y = (IMG[j].y * Tl[i].x - IMG[j].x * Tl[i].y);

  // Move center to around
  int of = (l / 2) % 2, st;
  if (of == local_y % 2)
    st = 1;
  else
    st = 0;
  if ((local_x - st) % 2 == 0) {
    CCG[i].x *= -1;
    CCG[i].y *= -1;
  }
}

// compute the avg of CCG in all templates
__global__ void add_CCG_to_sum(cufftComplex* CCG_sum, cufftComplex* CCG, int l, int N_tmp,
                               int block_id) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  // Area of rectangle, l^2
  int interval = l * l;
  int off = block_id * interval;

  // compute average & vairance
  for (int n = 0; n < N_tmp; n++) {
    float cur = CCG[n * interval + i].x / interval;
    CCG_sum[off + i].x += cur;
    CCG_sum[off + i].y += (cur * cur);
  }
}

__global__ void set_CCG_mean(cufftComplex* CCG_sum, int l, int N_tmp, int N_euler) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  float total_n = N_tmp * N_euler;

  float avg = CCG_sum[i].x / total_n;
  float var = sqrtf(CCG_sum[i].y / total_n - avg * avg);

  CCG_sum[i].x = avg;
  CCG_sum[i].y = var;
}

// update CCG val use avgeage & variance
__global__ void update_CCG(cufftComplex* CCG_sum, cufftComplex* CCG, int l, int block_id) {
  // On this function,block means subimage splitted from IMG, not block ON GPU
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  // Local id corresponding to splitted IMG
  int local_id = i % (l * l);
  int off = block_id * l * l;

  float avg = CCG_sum[off + local_id].x;
  float var = CCG_sum[off + local_id].y;

  float cur = CCG[i].x / l / l;
  CCG[i].x = var > 0 ? (cur - avg) / var : 0;
}

//"MAX" reduction for *odata : return max{odata[i]},i
//"SUM" reduction for *odata : return sum{odata[i]},sum{odata[i]^2}
__global__ void get_peak_and_SUM(cufftComplex* odata, float* res, int l, float d_m) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int tid = threadIdx.x;
  int image_size = l * l;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;

  sdata[tid] = odata[i].x;

  if (x < d_m / 4 || x > l - d_m / 4 || y < d_m / 4 || y > l - d_m / 4) sdata[tid] = 0;
  sdata[tid + blockDim.x] = local_id;
  sdata[tid + 2 * blockDim.x] = odata[i].x;
  sdata[tid + 3 * blockDim.x] = odata[i].x * odata[i].x;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      // find max
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        sdata[tid + blockDim.x] = sdata[tid + blockDim.x + s];
      }
      // sum of data[i] & data[i]^2
      sdata[tid + 2 * blockDim.x] += sdata[tid + 2 * blockDim.x + s];
      sdata[tid + 3 * blockDim.x] += sdata[tid + 3 * blockDim.x + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    res[blockIdx.x * 4] = sdata[0];
    res[blockIdx.x * 4 + 1] = sdata[blockDim.x];
    res[blockIdx.x * 4 + 2] = sdata[2 * blockDim.x];
    res[blockIdx.x * 4 + 3] = sdata[3 * blockDim.x];
  }
}

//"MAX" reduction for *odata : return max{odata[i]},i
__global__ void get_peak_pos(cufftComplex* odata, float* res, int l) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int tid = threadIdx.x;
  int image_size = l * l;
  int local_id = i % image_size;

  sdata[tid] = odata[i].x;
  sdata[tid + blockDim.x] = local_id;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      // find max
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        sdata[tid + blockDim.x] = sdata[tid + blockDim.x + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    res[blockIdx.x * 2] = sdata[0];
    res[blockIdx.x * 2 + 1] = sdata[blockDim.x];
  }
}

// CUFFT will enlarge VALUE to N times. Restore it
__global__ void scale(cufftComplex* data, size_t size, int l2) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  data[i].x /= l2;
  data[i].y /= l2;
}

__global__ void clear_image(cufftComplex* data) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i].x = 0;
  data[i].y = 0;
}

__global__ void clear_float(float* data) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i] = 0;
}

__global__ void Complex2float(float* f, cufftComplex* c, int nx, int ny) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx * ny) return;
  f[i] = c[i].x;
}

__global__ void float2Complex(cufftComplex* c, float* f, int nx, int ny) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx * ny) return;
  c[i].x = f[i];
  c[i].y = 0;
}

__global__ void do_phase_flip(cufftComplex* filter, Parameters para, int nx, int ny) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx * ny) return;

  int x = i % nx;
  int y = i / nx;
  float v = CTF_AST(x, (y + ny / 2) % ny, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff,
                    para.dfang, para.lambda, para.cs, para.ampconst, 0);

  filter[i].x *= v;
  filter[i].y *= v;
}

__global__ void ap2ri(cufftComplex* data) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  // ap2ri
  float tmp = data[i].x * sinf(data[i].y);
  data[i].x = data[i].x * cosf(data[i].y);
  data[i].y = tmp;
}

__global__ void ri2ap(cufftComplex* data, size_t size) {
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  // ri2ap
  float tmp = hypotf(data[i].x, data[i].y);
  if (data[i].x == 0 && data[i].y == 0)
    data[i].y = 0;
  else
    data[i].y = atan2(data[i].y, data[i].x);
  data[i].x = tmp;
}
