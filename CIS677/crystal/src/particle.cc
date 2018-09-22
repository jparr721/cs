#include <crystal/particle.hpp>

namespace crystal {
  Particle::Particle(int particle_num, bool caught, int x, int y) {
    this->x = x;
    this-> y = y;
    this->is_caught = false;
  }

  Particle::update(int x, int y, bool caught) {
    this->x = x;
    this->y = y;
    this->is_caught = caught;
  }

}// namespace crystal
