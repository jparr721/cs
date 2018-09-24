#include <iostream>
#include <omp.h>


int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: pi iterations" << std::endl;
    return EXIT_FAILURE;
  }

  long int iterations = atoi(argv[1]);

  double step = 1.0/ static_cast<double>(iterations);

  double x, sum = 0.0;

  #pragma omp parallel
  {
    #pragma omp for
    for (long int i = 1; i < iterations; i++) {
      #pragma omp critical
      {
        x = (i - 0.5) * step;
        sum += 4.0/(1.0 + x*x);
      }
    }
  }

  std::cout << "PI is " << sum * step << std::endl;

  return EXIT_SUCCESS;
}
