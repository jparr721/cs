#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

class PointSourcePollution {
  public:
    explicit PointSourcePollution();
    ~PointSourcePollution() = default;
    std::vector<double> diffuse(
        uint64_t cylinder_size,
        uint64_t diffusion_time,
        uint64_t contaminant_concentration);
    std::vector<std::vector<double>> diffuse2d(
        uint64_t pool_rows,
        uint64_t pool_cols,
        uint64_t diffusion_time,
        uint64_t contaminant_concentration,
        uint64_t leaks,
        bool multiple);
    void end(const std::vector<double> & data);
    void end2d(const std::vector<std::vector<double>>& data);
  private:
    double central_difference_theorem(double left, double right) const;
    double central_difference_theorem2d(double left, double right, double up, double down) const;
};

PointSourcePollution::PointSourcePollution() {}

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

void PointSourcePollution::end2d(const std::vector<std::vector<double>>& data) {
  std::ofstream payload;
  payload.open("output.txt");

  for (uint64_t i = 0; i < data.size(); ++i) {
    for (uint64_t j = 0; j < data[i].size(); ++j) {
      if (j != 0) {
        payload << " ";
      }

      payload << data[i][j];
    }
    payload << "\n";
  }
  payload.close();
}

double PointSourcePollution::central_difference_theorem2d(double left, double right, double up, double down) const {
  return (left + right + up + down) / 4;
}

std::vector<std::vector<double>> PointSourcePollution::diffuse2d(
    uint64_t pool_rows,
    uint64_t pool_cols,
    uint64_t diffusion_time,
    uint64_t contaminant_concentration,
    uint64_t leaks,
    bool multiple
    ) {
  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<std::tuple<uint64_t, uint64_t>> leak_locations;

  for (uint64_t i = 0; i < leaks; ++i) {
    std::uniform_int_distribution<uint64_t> leak_row(0, pool_rows);
    std::uniform_int_distribution<uint64_t> leak_col(0, pool_cols);
    std::tuple<uint64_t, uint64_t> val = std::make_tuple(leak_row(g), leak_col(g));
    leak_locations.push_back(val);
  }

  std::vector<std::vector<double>> pool(pool_rows, std::vector<double>(pool_cols, 0));
  std::vector<std::vector<double>> copy_pool(pool_rows, std::vector<double>(pool_cols, 0));
  std::vector<std::vector<double>> temp(pool_rows, std::vector<double>(pool_cols, 0));

  if (multiple) {
    for (uint64_t i = 0; i < leak_locations.size(); ++i) {
      auto row = std::get<0>(leak_locations[i]);
      auto col = std::get<1>(leak_locations[i]);

      pool[row][col] = contaminant_concentration;
      copy_pool[row][col] = contaminant_concentration;
    }
  } else {
    pool[0][0] = contaminant_concentration;
    copy_pool[0][0] = contaminant_concentration;
  }

  double left, right, up, down;

  for (uint64_t k = 0; k < diffusion_time; ++k) {
    for (uint64_t i = 0; i < pool_rows; ++i) {
      for (uint64_t j = 0; j < pool_cols; ++j) {
        left = j - 1 >= 0
          ? pool[i][j - 1]
          : pool[i][j];
        right = j + 1 <= pool_cols
          ? pool[i][j + 1]
          : pool[i][j];
        up = i - 1 >= 0
          ? pool[i - 1][j]
          : pool[i][j];
        down = i + 1 <= pool_rows
          ? pool[i + 1][j]
          : pool[i][j];
        copy_pool[i][j] = central_difference_theorem2d(left, right, up, down);
      }
    }
    temp = pool;
    pool = copy_pool;
    copy_pool = temp;
  }

  return pool;
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
  uint64_t pool_rows, pool_cols, leaks, cylinder_size, slice_location, diffusion_time, contaminant_concentration;
  bool multiple = false;

  if (argc < 5) {
    std::cerr << "usage: psp cylinder_size slice_location diffusion_time contaminant_concentration" << std::endl;
    std::cerr << "usage: pool_rows pool_cols leaks contaminant_concentration diffusion_time" << std::endl;
    return EXIT_FAILURE;
  }

  if (argc > 5) {
    pool_rows = atoi(argv[1]);
    pool_cols = atoi(argv[2]);
    leaks = atoi(argv[3]);
    if (leaks > 1) {
      multiple = true;
    }
    contaminant_concentration = atoi(argv[4]);
    diffusion_time = atoi(argv[5]);
  } else {
    cylinder_size = atoi(argv[1]);
    slice_location = atoi(argv[2]);
    diffusion_time = atoi(argv[3]);
    contaminant_concentration = atoi(argv[4]);
  }

  for (int i = 0; i < argc; ++i) {
    if (atoi(argv[i]) < 0) {
      std::cerr << "All inputs must be greater than 0" << std::endl;
      return EXIT_FAILURE;
    }
  }

  PointSourcePollution psp;
  if (argc == 5) {
    std::cout << "starting 1d diffusion..." << std::endl;
    std::vector<double> output = psp.diffuse(cylinder_size, diffusion_time, contaminant_concentration);
    std::cout << "Answer at slice location: " << slice_location << " is " << output[slice_location] << std::endl;
    std::cout << "Now visualizing results..." << std::endl;
    psp.end(output);
    system("python plot.py output.txt");
  } else {
    std::cout << "starting 2d diffusion" << std::endl;
    std::vector<std::vector<double>> out(pool_rows, std::vector<double>(pool_cols));
    out = psp.diffuse2d(pool_rows, pool_cols, diffusion_time, contaminant_concentration, leaks, multiple);
    std::cout << "Now visualizing results..." << std::endl;
    psp.end2d(out);
    system("python plot.py output.txt");
  }

  return EXIT_SUCCESS;
}
