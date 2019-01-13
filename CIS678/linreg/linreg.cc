#include <linreg/linreg.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

LinReg::LinReg(const std::string& filepath) {
  if (!read_infile(filepath, left, right)) {
    throw std::runtime_error("Error reading input file");
  }
}

void LinReg::print_vectors() {
  std::for_each(left.begin(), left.end(), [&](const auto v) {
    std::cout << "left: "<< v << std::endl;
  });

  std::for_each(right.begin(), right.end(), [&](const auto v) {
    std::cout << "right: "<< v << std::endl;
  });
}

std::pair<double, double> LinReg::trendline(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size()) {
    throw std::runtime_error("Vectors not the same size");
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

std::vector<double> LinReg::polynomial_trendline(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const int& order) {
  std::vector<double> coefficients;

  if (x.size() != y.size()) {
    throw std::runtime_error("Vectors not the same size");
  }

  size_t N = x.size();
  int n = order;
  int np1 = n + 1;
  int np2 = n + 2;
  int tnp1 = 2 * n + 1;
  double tmp;

  // X stores the values of sigma(xi^2n)
  std::vector<double> X(tnp1, 0);
  for (int i = 0; i < tnp1; ++i) {
    for (size_t j = 0u; j < N; ++j) {
      X[i] += std::pow(x[j], i);
    }
  }

  // A stores the final coefficients
  std::vector<double> A(np1);

  // B is an augmented matrix
  std::vector<std::vector<double>> B(np1, std::vector<double>(np2, 0));

  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      B[i][j] = X[i + j];
    }
  }

  std::vector<double> Y(np1, 0);
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      Y[i] += std::pow(x[j], i) * y[j];
    }
  }

  // Put Y as last column of B
  for (int i = 0; i <= n; ++i) {
    B[i][np1] = Y[i];
  }

  n += 1;
  int nm1 = n - 1;

  // Get pivot positions of B matrix
  for (int i = 0; i < n; ++i) {
    for (int k = i+1; k < n; ++k) {
      if (B[i][i] < B[k][i]) {
        for (int j = 0; j <= n; ++j) {
          tmp = B[i][j];
          B[i][j] = B[k][j];
          B[k][j] = tmp;
        }
      }
    }
  }

  // Perform the gaussian elimination
  for (int i = 0; i < n; ++i) {
    for (int k = i+1; k < n; ++k) {
      double t = B[k][i] / B[i][i];
      for (int j = 0; j <= n; ++j) {
        B[k][j] -= t * B[i][j];
      }
    }
  }

  // Back substitute up the call tree
  for (int i = nm1; i >= 0; --i) {
    A[i] = B[i][n];
    for (int j = 0; j < n; ++j) {
      if (j != i) {
        A[i] -= B[i][j] * A[j];
      }
    }
    A[i] /= B[i][i];
  }

  coefficients.resize(A.size());
  for (size_t i = 0u; i < A.size(); ++i) {
    coefficients[i] = A[i];
  }

  return coefficients;
}

bool LinReg::read_infile(
    const std::string& file,
    std::vector<double>& left,
    std::vector<double>& right) {
  std::ifstream in(file);

  if (!in) {
    std::cerr << "Error reading file path: " << file << std::endl;
    return false;
  }

  std::string line;
  bool on = false;
  while (std::getline(in, line)) {
    if (line.size() > 0) {
      if (line != "nan") {
        int i = 0;

        std::vector<double> temp;
        std::stringstream ss(line);

        while (ss >> i) {
          temp.push_back(i);

          if (ss.peek() == ',') { ss.ignore(); }
        }
        left.push_back(temp[0]);
        right.push_back(temp[1]);
        temp.erase(temp.begin());
      }
    }
  }

  in.close();
  return true;
}
