#include <cstdint>
#include <iostream>
#include <vector>

class PointSourcePollution {
  public:
    explicit PointSourcePollution(int cylinder_size) : simulation_space(simulation_space(cylinder_size, 0));
    ~PointSourcePollution() = default;
  private:
    std::vector<uint64_t> simulation_space;
    uint64_t heat_distribution;
    double central_difference_theorem(double left, double right);
    void diffuse(
        std::vector<uint64_t> *space,
        uint64_t cylinder_size,
        uint64_t slice_size,
        uint64_t slice_location,
        uint64_t diffusion_time,
        uint64_t contaminant_concentration);
};

void PointSourcePollution::diffuse(
    std::vector<uint64_t> *space,
    uint64_t cylinder_size,
    uint64_t slice_size,
    uint64_t slice_location,
    uint64_t diffusion_time,
    uint64_t contaminant_concentration
    ) {
  for (uint64_t i = 0; i < diffusion_time; ++i) {

  }
}

double PointSourcePollution::central_difference_theorem(double left, double right) {
  return (left + right) / 2.0;
}

int main(int argc, char** argv) {
  uint64_t cylinder_size, slice_size, slice_location, diffusion_time, contaminant_concentration;

  if (argc < 6) {
    std::cerr << "usage: psp cylinder_size slice_size slice_location diffusion_time" << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < argc; ++i) {
    if (atoi(argv[i]) < 0) {
      std::cerr << "All inputs must be greater than 0" << std::endl;
      return EXIT_FAILURE;
    }
  }

  cylinder_size = atoi(argv[1]);
  slice_size = atoi(argv[2]);
  slice_location = atoi(argv[3]);
  diffusion_time = atoi(argv[4]);
  contaminant_concentration = atoi(argv[5]);
}
