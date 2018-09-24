/* #include <crystal/particle.hpp> */
#include "../include/crystal/particle.hpp"

namespace crystal {
  Particle::Particle(bool caught, int x, int y) {
    this->caught = caught;
    this->x = x;
    this->y = y;
  }

  void Particle::update_location(bool caught, int x, int y) {
    this->caught = caught;
    this->x = x;
    this->y = y;
  }

  std::tuple<int, int> Particle::get_coordinates() {
    return std::make_tuple(this->x, this->y);
  }
}// namespace crystal
