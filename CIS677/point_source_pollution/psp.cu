#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

const int BLOCK_SIZE = 1024;

class PointSourcePollution {
  public:
    PointSourcePollution() = default;
    ~PointSourcePollution() = default;
    void end(const double* data);
}

void PointSourcePollution::end(const std::vector<double> data) {
  std::ofstream payload;
  payload.open("output.txt");

  for (uint64_t i = 0; i < data.size(); ++i) {
    if (i != 0) {
      payload << " ";
    }

    paylaod << data[i];
  }

  payload.close();
}

__device__
void central_difference_theorem(
    double left,
    double, right,
    double* out
    ) {
  return (left + right) / 2.0;
}

__global__
void make_array(
    double* cylinder,
    double* copy_cylinder,
    uint64_t num_slices
    uint64_t concentration
    ) {
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (i < num_slices) {
    if (i == 0) {
      cylinder[i] = concentration;
    } else {
      cylinder[i] = 0.0;
    }
  }
}

__global__
void diffuse(
    double* cylinder,
    double* copy_cylinder,
    double* output,
    uint64_t cylinder_size,
    uint64_t num_slices,
    uint64_t diffusion_time,
    uint64_t contaminant_concentration
    ) {
  double left, right;
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
}


int main(int argc, char** argv) {
  uint64_t cylinder_size, num_slices, slice_location, diffusion_time, contaminant_concentration;

  if (argc < 6) {
    std::cerr << "usage: psp cylinder_size num_slices slice_location diffusion_time contaminant_concentration" << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < argc; ++i) {
    if (atoi(argv[i]) < 0) {
      std::cerr << "All inputs must be greater than 0" << std::endl;
      return EXIT_FAILURE;
    }
  }

  PointSourcePollution psp();
  cylinder_size = atoi(argv[1]);
  num_slices = atoi(argv[2]);
  slice_location = atoi(argv[3]);
  diffusion_time = atoi(argv[4]);
  contaminant_concentration = atoi(argv[5]);
  cudaError_t e;
  double* cylinder, copy_cylinder, output;

  e = cudaMalloc((void**) &cylinder, cylinder_size * sizeof(double));
  if (e != cudaSuccess) {
    std::cerr << "Error performing cuda malloc for cylinder" << std::endl;
    return EXIT_FAILURE;
  }

  e = cudaMalloc((void**) &copy_cylinder, cylinder_size * sizeof(double));
  if (e != cudaSuccess) {
    std::cerr << "Error performing cuda malloc for copy cyliner" << std::endl;
    return EXIT_FAILURE;
  }

  e = cudaMalloc((void**) &output, cylinder_size * sizeof(double));
  if (e != cudaSuccess) {
    std::cerr << "Erro performaing cuda malloc for output" << std::endl;
    return EXIT_FAILURE;
  }

  const uint64_t GRID_SIZE = cylinder_size / BLOCK_SIZE;
  make_array<<<GRID_SIZE, BLOCK_SIZE>>>(cylinder, copy_cylinder, num_slices, contaminaint_concentration);
  diffuse<<<GRID_SIZE, BLOCK_SIZE>>>(
      cylinder,
      copy_cylinder,
      output,
      cylinder_size,
      num_slices,
      diffusion_time,
      contaminant_concentration);

  int elements_in_output = sizeof(output) / sizeof(output[0]);
  std::vector<double> data(output, output + cylinder_size);
  std::cout << "Answer at slice location: " << slice_location << " is " << output[slice_location] << std::endl;
  std::cout << "Now visualizing results..." << std::endl;
  psp.end(data);

  e = cudaFree(output);
  if (e != cudaSuccess) {
    std::cerr << "Error performing memory free on output" << std::endl;
    return EXIT_FAILURE;
  }

  e = cudaFree(cylinder);
  if (e != cudaSuccess) {
    std::cerr << "Error performing memory free on cylinder" << std::endl;
    return EXIT_FAILURE;
  }

  e = cudaFree(copy_cylinder);
  if (e != cudaSuccess) {
    std::cerr << "Error perofmring memory free on copy_cylinder" << std::endl;
    return EXIT_FAILURE;
  }


  system("python plot.py");
  return EXIT_SUCCESS;
}
