#pragma once

#include <vector>

namespace stats {
class StatisticallyRandomSubsets {
public:
  int partition(std::vector<int> & arr, int low, int high);
  std::vector<int> sort(std::vector<int> & unsorted_vector, int low, int high);
  std::vector<int> generate(int k, const std::vector<int> & n);
};
} // namespace stats
