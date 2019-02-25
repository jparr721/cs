#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <string>
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

  template<typename TK, typename TV>
  std::vector<TK> extract_keys(std::unordered_map<TK, TV> const& input_map) {
    std::vector<TK> retval;
    for (auto const& element : input_map) {
      retval.push_back(element.first);
    }
    return retval;
  }

  template<typename TK, typename TV>
  std::vector<TV> extract_values(std::unordered_map<TK, TV> const& input_map) const {
    std::vector<TV> retval;
    for (auto const& element : input_map) {
      retval.push_back(element.second);
    }
    return retval;
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

  void print_vector(const std::vector<std::vector<std::string>>& the_vector) {
    for (const auto& row : the_vector) {
      for (const auto& col : row) {
        std::cout << col << ", ";
      }

      std::cout << std::endl;
    }
  }
} // namespace util
  using probability_map =
    std::unordered_map<std::string, double>;

class Arbol {
  public:
    Arbol(const std::map<std::string, std::vector<std::string>> value_ranges) : value_ranges_(value_ranges) {};

    void fit();
  private:
    double entropy(const std::string& k);
    double gain(const double S, const double a);

    probability_map class_probabilities_;
    probability_map entropies_;
    std::map<std::string, std::vector<std::string>> value_ranges_;
};
} // namespace arbol
