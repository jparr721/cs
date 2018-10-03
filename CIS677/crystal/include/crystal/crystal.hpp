#ifndef CRYSTAL_CRYSTAL_HPP
#define CRYSTAL_CRYSTAL_HPP

#include <tuple>
#include <vector>
#include <cstdint>

namespace crystal {
  class Crystal {
  public:
    int64_t ROWS;
    int64_t COLS;
    int64_t SIMULATION_SIZE;
    int64_t MAX_MOVES;
    int64_t CENTER;

    Crystal(int64_t, int64_t);
    ~Crystal() = default;

    void Run(int64_t);
    void end_simulation(const std::vector<std::vector<int>>&);
    std::tuple<int64_t, int64_t> insert_particle(const std::vector<std::vector<int>>&, int);
    void random_walk(int64_t&, int64_t&, std::vector<std::vector<int>>&);
    bool collision(int64_t, int64_t, const std::vector<std::vector<int>>&);
    void print(const std::vector<std::vector<int>>&);
  };
} // namespace crystal

#endif
