#include <cmath>
#include <stdexcept>
#include "arbol.h"

namespace arbol {
  void Arbol::entropy() {
    for (const auto& k : classes_) {
      double probability{class_probabilities_[k]};
      entropies_[k] = -probability * std::log2(probability);
    }
  }
} // namespace arbol

int main(int argc, char** argv) {
  return EXIT_SUCCESS;
}
