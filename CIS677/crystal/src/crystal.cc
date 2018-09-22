#include <crystal/crystal.hpp>

namespace crystal {
  Crystal::Crystal(std::vector<int> board_size) {
    int rows = board_size[0];
    int columns = board_size[1];

    this->simulation_matrix(
        rows,
        std::vector<int>(columns));
  }

  void Crystal::Run(int iterations) {
  }

  int Crystal::get_particles() {
    return this->particles;
  }
}// namespace crystal
