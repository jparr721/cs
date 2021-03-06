#ifndef ARBOL_H_
#define ARBOL_H_

#include <Eigen/Dense>
#include <fstream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace arbol {
namespace util {
  template<typename M>
  M load_csv(const std::string& path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<
      const Eigen::Matrix<
      typename M::Scalar,
               M::RowsAtCompileTime,
               M::ColsAtCompileTime,
               Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
  }

  template<typename MapType, typename TK>
  std::vector<TK> extract_keys(const MapType& input_map) {
    std::vector<TK> retval;
    for (const auto& element : input_map) {
      retval.push_back(element.first);
    }
    return retval;
  }

  template<typename MapType, typename TV>
  std::vector<TV> extract_values(const MapType& input_map) {
    std::vector<TV> retval;
    for (const auto& element : input_map) {
      retval.push_back(element.second);
    }
    return retval;
  }

  template <typename T>
  std::string vec_to_string(const std::vector<T>& values) {
    std::string retval;
    for (const auto& val : values) {
      retval += val + ",";
    }

    return retval;
  }

  template <typename T>
  void print_2d_vector(const std::vector<std::vector<T>>& values) {
    for (const auto& val : values) {
      for (const auto& v : val) {
        std::cout << v << ",";
      }
      std::cout << std::endl;
    }
  }

  std::vector<std::vector<std::string>> load_non_numeric(const std::string& path) {
    std::ifstream indata;
    indata.open(path);

    std::vector<std::vector<std::string>> values;
    std::string line;
    while (std::getline(indata, line)) {
      std::vector<std::string> line_data;
      std::stringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) {
        line_data.push_back(cell);
      }

      values.push_back(line_data);
    }

    return values;
  }

  bool is_integer(const std::string & s) {
     if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;

     char * p;
     strtol(s.c_str(), &p, 10);

     return (*p == 0);
  }

  std::vector<std::string> split(std::string line) {
    std::vector<std::string> result;
    std::istringstream iss(line);
    for (std::string line; iss >> line;)
      result.push_back(line);

    return result;
  }
} // namespace util

using probability_map =
  std::unordered_map<std::string, double>;

struct data_frame {
  data_frame(
      int nt,
      std::vector<std::string> t,
      int na,
      std::map<std::string, std::vector<std::string>> a,
      int ne,
      std::vector<std::vector<std::string>> av) :
    num_targets(nt), targets(t), num_attributes(na), attributes(a), num_examples(ne), attribute_values(av) {};

  int num_targets;
  std::vector<std::string> targets;

  int num_attributes;
  std::map<std::string, std::vector<std::string>> attributes;

  int num_examples;
  std::vector<std::vector<std::string>> attribute_values;
};

template<class T>
struct Node<T> {
  int index;
  int tree_index;

  bool isLeaf;

  std::string label;

  T value;

  std::vector<int> children;

  Node() {
    this.is_leaf = false;
  }
};

using Tree = std::vector<Node<std::string>>;

class Arbol {
  public:
    std::unique_ptr<data_frame> make_dataframe(const std::string& path);
    void fit(std::shared_ptr<data_frame> data);
  private:
    void calculate_total_entropy(std::shared_ptr<data_frame> data);

    int dfs(const std::vector<std::string>& row, int idx);

    double entropy(const std::string& k);
    double gain(const double S, const double a);

    std::string guess(const std::vector<std::string>& row);

    double total_entropy{0.0};
    probability_map class_probabilities_;
    probability_map entropies_;
    Tree decision_tree;
};
} // namespace arbol

#endif // ARBOL_H_
