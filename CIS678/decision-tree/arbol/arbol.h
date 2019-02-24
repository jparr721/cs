#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace arbol {
namespace util {
  template<typename M>
  M load_csv (const std::string & path) {
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
} // namespace util
  using probability_map =
    std::unordered_map<std::string, double>;

class Arbol {
  public:
    void entropy();
  private:
    probability_map class_probabilities_;
    probability_map entropies_;
    std::vector<std::string> classes_;
};
} // namespace arbol
