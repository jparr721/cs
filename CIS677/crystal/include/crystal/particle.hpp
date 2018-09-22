#ifndef CRYSTAL_PARTICLE_HPP
#define CRYSTAL_PARTICLE_HPP

#include <vector>

namespace crystal {
  class Particle {
    public:
      int x;
      int y;
      Particle(int particle_num, int x, int y);
      ~Particle() = default;
      Particle(Particle &&) = default;

      void update(int x, int y, bool caught);

      // For debugging
      std::vector<int> get_coordinates();

    private:
      bool is_caught;
      bool check_caught();

  };

} //namespace crystal

#endif
