/* #include <linreg/linreg.h> */
#include "./include/linreg.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <utility>

LinReg::LinReg(const std::string& filepath) {
  if (!read_infile(filepath, lines)) {
    throw std::runtime_error("Error reading input file");
  }
}

std::pair<double, double> tendline(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("Vectors not the same size");
  }
  const int n = x.size();
  const double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
  const double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
  const double sum_x_sq = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
  const double sum_xy_pair = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
  const double slope = ((n * sum_xy_pair) - (sum_x * sum_y)) / ((n * sum_x_sq) - sum_x * sum_x);
  const double intercept = (sum_y - (slope * sum_x)) / n;

  return std::make_pair(slope, intercept);
}

bool read_infile(const std::string& file, std::vector<double>& lines) {
  std::ifstream in(file);

  if (!in) {
    std::cerr << "Error reading file path: " << file << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.size() > 0) {
      if (line != "nan") {
        lines.push_back(std::stod(line));
      }
    }
  }

  in.close();
  return true;
}
