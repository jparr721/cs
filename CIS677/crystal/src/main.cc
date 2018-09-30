#include <crystal/crystal.hpp>
#include <iostream>
#include "omp.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "usage: ./crystal number_of_particles simulation_size" << std::endl;
    return EXIT_FAILURE;
  }

  int particles = atoi(argv[1]);
  int simulation_size = atoi(argv[2]);

  crystal::Crystal crystal(particles, simulation_size);

  crystal.Run(particles);

  return EXIT_SUCCESS;
}
