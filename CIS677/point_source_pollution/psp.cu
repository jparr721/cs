#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

const int BLOCK_SIZE = 1024;

class PointSourcePollution {
  public:
    PointSourcePollution() = default;
    ~PointSourcePollution() = default;
    void end(const double* data, uint64_t cylinder_size);
};

void PointSourcePollution::end(const double* data, uint64_t cylinder_size) {
  std::ofstream payload;
  payload.open("output.txt");

  for (uint64_t i = 0; i < cylinder_size; ++i) {
    if (i != 0) {
      payload << " ";
    }

    payload << data[i];
  }

  payload.close();
}

__device__
void central_difference_theorem(
    double left,
    double right,
    double& out
    ) {
  out = (left + right) / 2.0;
}

__global__
void diffuse(
    double* cylinder,
    double* copy_cylinder,
    double* temp,
    uint64_t cylinder_size,
    uint64_t diffusion_time,
    uint64_t contaminant_concentration
    ) {
  double left, right, cdt_out;
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (i < cylinder_size) {
    if (i > 0)
      left = cylinder[i - 1];
    else
      left = cylinder[i];
    right = cylinder[i + 1];

    central_difference_theorem(left, right, cdt_out);
    cylinder[i] = cdt_out;
    temp = cylinder;
    cylinder = copy_cylinder;
    copy_cylinder = temp;
  }
}


int main(int argc, char** argv) {
  uint64_t cylinder_size, slice_location, diffusion_time, contaminant_concentration;

  if (argc < 5) {
    std::cerr << "usage: psp cylinder_size slice_location diffusion_time contaminant_concentration" << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < argc; ++i) {
    if (atoi(argv[i]) < 0) {
      std::cerr << "All inputs must be greater than 0" << std::endl;
      return EXIT_FAILURE;
    }
  }

  PointSourcePollution psp;
  cylinder_size = atoi(argv[1]);
  slice_location = atoi(argv[2]);
  diffusion_time = atoi(argv[3]);
  contaminant_concentration = atoi(argv[4]);
  cudaError_t e;
  double *cylinder, *copy_cylinder, *temp;

  cudaMallocManaged(&cylinder, cylinder_size * sizeof(double));
  cudaMallocManaged(&copy_cylinder, cylinder_size * sizeof(double));
  cudaMallocManaged(&temp, cylinder_size * sizeof(double));

  // init our arrays
  for (int i = 0; i < cylinder_size; ++i) {
    if (i == 0) {
      cylinder[i] = contaminant_concentration;
      copy_cylinder[i] = contaminant_concentration;
    } else {
      cylinder[i] = 0.0;
      copy_cylinder[i] = 0.0;
    }
  }
  std::cout << cylinder[0] << copy_cylinder[0] << std::endl;

  e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(e) << std::endl;
    return EXIT_FAILURE;
  }

  const uint64_t GRID_SIZE = ceil(cylinder_size / static_cast<double>(BLOCK_SIZE));
  for (int i = 0; i < diffusion_time; ++i) {
    diffuse<<<GRID_SIZE, BLOCK_SIZE>>>(
        cylinder,
        copy_cylinder,
        temp,
        cylinder_size,
        diffusion_time,
        contaminant_concentration);
  }
  cudaDeviceSynchronize();

  e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cerr << "Error2: " << cudaGetErrorString(e) << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Answer at slice location: " << slice_location << " is " << cylinder[slice_location] << std::endl;
  std::cout << "Now visualizing results..." << std::endl;
  psp.end(cylinder, cylinder_size);

  cudaFree(cylinder);
  cudaFree(copy_cylinder);

  e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::cerr << cudaGetErrorString(e) << std::endl;
    return EXIT_FAILURE;
  }


  system("python plot.py");
  return EXIT_SUCCESS;
}
