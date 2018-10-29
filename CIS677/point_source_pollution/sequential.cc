#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

class PointSourcePollution {
  public:
    explicit PointSourcePollution(uint64_t cylinder_size);
    ~PointSourcePollution() = default;
    std::vector<double> diffuse(
        uint64_t cylinder_size,
        uint64_t diffusion_time,
        uint64_t contaminant_concentration);
    void end(const std::vector<double> & data);
  private:
    std::vector<uint64_t> simulation_space;
    double central_difference_theorem(double left, double right) const;
};

PointSourcePollution::PointSourcePollution(uint64_t cylinder_size) : simulation_space(cylinder_size, 0) {}

void PointSourcePollution::end(const std::vector<double>& data) {
  std::ofstream payload;
  payload.open("output.txt");

  for (uint64_t i = 0; i < data.size(); ++i) {
    if (i != 0) {
      payload << " ";
    }
    payload << data[i];
  }

  payload.close();
}

std::vector<double> PointSourcePollution::diffuse(
    uint64_t cylinder_size,
    uint64_t diffusion_time,
    uint64_t contaminant_concentration
    ) {
  std::vector<double> cylinder(cylinder_size, 0);
  std::vector<double> copy_cylinder(cylinder_size, 0);
  double left, right;

  cylinder[0] = contaminant_concentration;
  copy_cylinder[0] = contaminant_concentration;

  for (uint64_t i = 1; i < diffusion_time; ++i) {
    for (uint64_t j = 1; j < cylinder_size; ++j) {
      left = cylinder[j - 1];
      right = cylinder[j + 1];

      copy_cylinder[j] = this->central_difference_theorem(left, right);
    }
    std::vector<double> temp(cylinder);
    cylinder = copy_cylinder;
    copy_cylinder = temp;
  }

  return cylinder;
}

double PointSourcePollution::central_difference_theorem(double left, double right) const {
  return (left + right) / 2.0;
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

  cylinder_size = atoi(argv[1]);
  slice_location = atoi(argv[2]);
  diffusion_time = atoi(argv[3]);
  contaminant_concentration = atoi(argv[4]);

  PointSourcePollution psp(cylinder_size);
  std::vector<double> output = psp.diffuse(cylinder_size, diffusion_time, contaminant_concentration);
  std::cout << "Answer at slice location: " << slice_location << " is " << output[slice_location] << std::endl;
  std::cout << "Now visualizing results..." << std::endl;
  psp.end(output);
  system("python plot.py");

  return EXIT_SUCCESS;
}
