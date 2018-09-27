#include <crystal/crystal.hpp>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: ./crystal number_of_particles" << std::endl;
    return EXIT_FAILURE;
  }

  crystal::Crystal crystal;

  crystal.Run();

  return EXIT_SUCCESS;
}
