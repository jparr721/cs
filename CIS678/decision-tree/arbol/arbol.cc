#include <cmath>
#include <iostream>
#include <stdexcept>
#include "arbol.h"

namespace arbol {
  double gain(const double entropy) {
    return 0.0;
  }

  double Arbol::entropy(const std::string& k) {
    double probability{class_probabilities_[k]};
    entropies_[k] = -probability * std::log2(probability);
    return 0.0;
  }
} // namespace arbol

int main(int argc, char** argv) {
  auto csv = arbol::util::load_csv(std::string(argv[1]));
  return EXIT_SUCCESS;
}
