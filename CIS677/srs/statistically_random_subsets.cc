#include "statistically_random_subsets.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>

namespace stats {
int StatisticallyRandomSubsets::partition(std::vector<int> & arr, int low, int high) {
  int pivot = arr[high];

  // Our artifical "wall"
  int i = low - 1;

  for (int j = low; j <= high - 1; ++j) {
    if (arr[j] <= pivot) {
      i++;
      std::iter_swap(arr.begin() + i, arr.begin() + j);
    }
  }

  std::iter_swap(arr.begin() + (i + 1), arr.begin() + high);

  return i + 1;
}

std::vector<int> StatisticallyRandomSubsets::sort(std::vector<int> & unsorted_vector, int low, int high) {
  if (low < high) {
    int p = partition(unsorted_vector, low, high);

    sort(unsorted_vector, low, p - 1);
    sort(unsorted_vector, p + 1, high);
  }

  return unsorted_vector;
}

std::vector<int> StatisticallyRandomSubsets::generate(int k, const std::vector<int> & n) {
  std::vector<int> random_list(n);
  std::random_device rd;

  // Mersaine Twister Pseudo random number genrator
  // This is an optimized random number generator in the stl
  std::mt19937 g(rd());

  std::shuffle(random_list.begin(), random_list.end(), g);
  random_list.resize(k);

  return random_list;
}
} // namespace stats

int main() {
  stats::StatisticallyRandomSubsets srs;

  std::vector<int> n;
  n.reserve(500);
  int k = 50;
  for (int i = 0; i < 500; ++i) {
    n.push_back(i);
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::vector<int> output = srs.generate(k, n);
  // Custom sort takes ~30ms longer
  /* output = srs.sort(output, 0, output.size() - 1); */

  // Faster sort option
  std::sort(output.begin(), output.end());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::copy(output.begin(), output.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n\nRunning Times" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "ms" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "ns" << std::endl;

  return 0;
}
