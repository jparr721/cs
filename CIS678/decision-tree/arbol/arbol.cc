#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "arbol.h"

namespace arbol {
  std::unique_ptr<data_frame> Arbol::make_dataframe(const std::string& path) {
    std::ifstream indata;
    // Open our data file
    indata.open(path);

    // Record a line
    std::string line;
    std::vector<std::string> lines;

    // Initialize our values to -1 to check for errors
    int num_targets{-1}, num_attributes{-1}, num_examples{-1};

    // Store lines in vector
    while (std::getline(indata, line)) {
      std::stringstream line_stream(line);
      lines.push_back(line_stream.str());
    }

    num_targets = std::stoi(lines[0]);
    num_attributes = std::stoi(lines[2]);
    num_examples = std::stoi(lines[2 + num_attributes]);


    auto targets = util::split(lines[1]);
    std::map<std::string, std::vector<std::string>> attributes;

    for (int i = 0; i < num_attributes; ++i) {
      int idx = i + 4;
      auto vals = util::split(lines[idx]);
      std::vector<std::string> sub(vals.begin() + 2, vals.end());;
      attributes[vals[0]] = sub;
    }

    std::vector<std::vector<std::string>> attribute_values;
    for (uint i = num_examples + 1; i < lines.size(); ++i) {
      auto vals = util::split(lines[i]);
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

  double Arbol::gain(const double S, const double a) {
    return S - a;
  }

  double Arbol::entropy(const std::string& k) {
    double probability{class_probabilities_[k]};
    entropies_[k] = -probability * std::log2(probability);
    return 0.0;
  }

  int Arbol::dfs(const std::vector<std::string>& row, int idx) {
    return 0;
  }

  std::string Arbol::guess(const std::vector<std::string>& row) {
    std::string label = "";
    int leaf_node = dfs(row, 0);

    return leaf_node != -1 ? decision_tree[leaf_node].label : "fail";
  }

  void Arbol::calculate_total_entropy(std::shared_ptr<data_frame> data) {
    const auto dataset = data->attribute_values;
    std::unordered_map<std::string, double> occurances;

    // Sum the classes
    for (const auto& row : dataset) {
      ++occurances[row[row.size() - 1]];
    }

    const auto vals = util::extract_values<std::unordered_map<std::string, double>, double>(occurances);
    const auto total = std::accumulate(vals.begin(), vals.end(), 0.0);

    double final_entropy = (occurances.begin()->second/total) * std::log2(occurances.begin()->second/total);
    for (auto it = std::next(occurances.begin(), 1); it != occurances.end(); ++it) {
      final_entropy -= (it->second/total * std::log2(it->second / total));
    }

    total_entropy = final_entropy;
  }

  void Arbol::fit(std::shared_ptr<data_frame> data) {

  }
} // namespace arbol

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: arbol <FILE_PATH>" << std::endl;
  }
  arbol::Arbol a;
  auto csv = a.make_dataframe(argv[1]);
  return EXIT_SUCCESS;
}
