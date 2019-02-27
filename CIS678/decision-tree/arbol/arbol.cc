#include <cmath>
#include <iostream>
#include <stdexcept>
#include "arbol.h"

namespace arbol {
  std::unique_ptr<data_frame> Arbol::make_dataframe(const std::string& path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<std::string> lines;
    uint idx = 0;
    int num_targets{-1}, num_attributes{-1}, num_examples{-1};

    // Store lines in vector
    while (std::getline(indata, line)) {
      std::stringstream line_stream(line);
      lines.push_back(line_stream.str());
    }

    num_targets = std::stoi(lines[0]);
    num_attributes = std::stoi(lines[2]);
    num_examples = std::stoi(lines[2 + num_attributes]);


    auto targets = split(lines[1]);
    std::map<std::string, std::vector<std::string>> attributes;

    for (int i = num_attributes + 1; i < num_examples; ++i) {
      auto vals = split(lines[i]);
      std::vector<std::string> sub;
      attributes[vals[0]] = (&lines[i][3], &lines[i][lines[i].size() - 1]);
    }

    std::vector<std::vector<std::string>> attribute_values;
    for (int i = num_exampes + 1; i < lines.size(); ++i) {
      auto vals = split(lines[i]);
      attribute_values.push_back(vals);
    }

    auto ptr = std::make_unique<data_frame>(
        num_targets,
        targets,
        num_attributes,
        attributes,
        num_examples,
        attribute_values);

    return ptr;
  }

  // This will calculate the gain sum for a class k
  double sum_attributes(const std::string& k) {
    // Get all of our possible values a class can be
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
    std::vector<std::string> labels = data_[0];
    auto class_label_name = labels[labels.size() - 1];

    // Remove the class label heading
    labels.pop_back();
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
    // First, we sum our other class labels
    for (int j = 0; j < labels.size(); ++j) {
      for (int i = 1; i < data.size(); ++i) {
        ++feature_probabilities_[labels[j]][data[i][j]];
      }
    }
  }

} // namespace arbol

int main(int argc, char** argv) {
  std::string arg = argv[1];
  auto csv = arbol::util::load_non_numeric(arg);
  arbol::util::print_vector(csv);
  return EXIT_SUCCESS;
}
