#include <cmath>
#include <iostream>
#include <stdexcept>
#include "arbol.h"

namespace arbol {
  double sum_attributes(const std::string& k) {
    auto val_opts = value_ranges[k];
  }

  double Arbol::gain(const double S, const double a) {
    return S - a;
  }

  double Arbol::entropy(const std::string& k) {
    double probability{class_probabilities_[k]};
    entropies_[k] = -probability * std::log2(probability);
    return 0.0;
  }

  void Arbol::fit(const std::vector<std::vector<std::string>> data) {
    const std::vector<std::string> labels = data_[0];
    auto class_label_name = labels[labels.size() - 1];
    auto total_entries_count = data.size();

    // First, we want to get the overall probability of each class label
    for (uint i = 1; i < data.size(); ++i) {
      ++class_probabilities_[data[i][labels.size() - 1]];
    }

    // Now, get our true probabilities of each label
    for (const auto& proba : class_probabilities_) {
      proba->second = (double)proba->second / (double)total_entries_count;
    }

    // Now, calculate the entropy of our labels
    double S{0.0};

    auto keys = util::extract_keys<std::string, double>(class_probabilities_);
    S = entropy(keys[0]);

    // Now we sum the rest into S.
    for (int i = 1; i < keys.size(); ++i) { S -= entropy(keys[i]); }

    // With S as a now static value, we can move on calculating mad gains
  }

} // namespace arbol

int main(int argc, char** argv) {
  std::string arg = argv[1];
  auto csv = arbol::util::load_non_numeric(arg);
  arbol::util::print_vector(csv);
  return EXIT_SUCCESS;
}
