#include <cstdint>
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
void make_array(
    double* cylinder,
    double* copy_cylinder,
    uint64_t cylinder_size,
    uint64_t concentration
    ) {
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (i < cylinder_size) {
    if (i == 0) {
      cylinder[i] = concentration;
      copy_cylinder[i] = concentration;
    } else {
      cylinder[i] = 0.0;
      copy_cylinder[i] = 0.0;
    }
  }
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
    for (int i = 1; i < diffusion_time; ++i) {
      for (int j = 1; j < cylinder_size; ++j) {
        left = cylinder[j - 1];
        right = cylinder[j + 1];

        central_difference_theorem(left, right, cdt_out);
        cylinder[j] = cdt_out;
      }
      temp = cylinder;
      cylinder = copy_cylinder;
      copy_cylinder = temp;
    }
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

  e = cudaMalloc((void**) &temp, cylinder_size * sizeof(double));
  if (e != cudaSuccess) {
    std::cerr << "Error performing cuda malloc for temp cylinder" << std::endl;
    return EXIT_FAILURE;
  }

  const uint64_t GRID_SIZE = cylinder_size / BLOCK_SIZE;
  make_array<<<GRID_SIZE, BLOCK_SIZE>>>(cylinder, copy_cylinder, cylinder_size, contaminant_concentration);
  diffuse<<<GRID_SIZE, BLOCK_SIZE>>>(
      cylinder,
      copy_cylinder,
      temp,
      cylinder_size,
      diffusion_time,
      contaminant_concentration);

  std::cout << "Answer at slice location: " << slice_location << " is " << cylinder[slice_location] << std::endl;
  std::cout << "Now visualizing results..." << std::endl;
  psp.end(cylinder, cylinder_size);

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
