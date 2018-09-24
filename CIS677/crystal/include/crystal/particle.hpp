#ifndef CRYSTAL_PARTICLE_HPP
#define CRYSTAL_PARTICLE_HPP

#include <tuple>

namespace crystal {
  class Particle {
    public:
      int x;
      int y;
      bool caught;
      Particle(bool, int, int);
      ~Particle() = default;

      void update_location(bool, int, int);

      // For debugging
      std::tuple<int, int> get_coordinates();
  };

} //namespace crystal

#endif
