#include <algorithm>
#include <chrono>
#include <crystal/crystal.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <omp.h>

namespace crystal {
  auto read_simulation_space = [](
      const int64_t x,
      const int64_t y,
      const std::vector<std::vector<int>>& simulation_space)->int {
    int val = 0;
    #pragma omp atomic read
    val = simulation_space[x][y];
    return val;
  };

  auto write_simulation_space = [](
      const int64_t x,
      const int64_t y,
      std::vector<std::vector<int>>& simulation_space, int val)->void {
    #pragma omp atomic write
    simulation_space[x][y] = val;
  };

  auto read_radius = [](const int& radius)->int {
    int val = 0;
    #pragma omp atomic read
    val = radius;
    return val;
  };

  auto write_radius = [](int& radius, int val)->void {
    #pragma omp atomic write
    radius = val;
  };

  Crystal::Crystal(int64_t particles, int64_t simulation_size) {
    this->SIMULATION_SIZE = simulation_size;
    this->ROWS = simulation_size;
    this->COLS = simulation_size;
    this->MAX_MOVES = 4 * simulation_size;
    this->CENTER = simulation_size / 2;
  }

  void Crystal::Run(int64_t particles) {
    int radius = 0;
    std::vector<std::vector<int>> simulation_space(
        this->ROWS,
        std::vector<int>(this->COLS));

    // Place our point in the center of the board
    simulation_space[this->CENTER][this->CENTER] = 1;
    omp_set_num_threads(16);

    auto now = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
      std::cout << omp_get_num_threads() << std::endl;
      for (int i = 0; i < particles; ++i) {
        if (read_radius(radius) >= this->SIMULATION_SIZE / 2) {
          break;
        }

        const auto point_location = this->insert_particle(simulation_space, radius);
        const int64_t x = std::get<0>(point_location);
        const int64_t y = std::get<1>(point_location);
        this->random_walk(x, y, simulation_space);

        if (x >= 0 && x < this->SIMULATION_SIZE && y >= 0 && y < this->SIMULATION_SIZE) {
          const int distance = std::max(std::abs(this->CENTER - x), std::abs(this->CENTER - y));

          if (distance > read_radius(radius)) {
            write_radius(radius, distance);
          }
        }
      }

      this->end_simulation(simulation_space);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - now).count() << "s" << std::endl;
  }

  void Crystal::end_simulation(const std::vector<std::vector<int>> &simulation_space) {
    std::ofstream the_goods;

    the_goods.open("output.txt");

    for (const auto& i : simulation_space) {
      for (const auto& j : i) {
        int current = 0;

        if (j == 1) {
          current = 1;
        }

        if (j != 0) {
          the_goods << " ";
        }

        the_goods << current;
      }
      the_goods << "\n";
    }

    the_goods.close();
  }

  void Crystal::print(const std::vector<std::vector<int>>& simulation_space) {
    for (uint64_t i = 0; i < this->SIMULATION_SIZE; ++i) {
      for (uint64_t j = 0; j < this->SIMULATION_SIZE; ++j) {
        int current = simulation_space[i][j];

        std::cout << current << ", " << std::endl;
      }

      std::cout << "\n" << std::endl;
    }
  }

  void Crystal::random_walk(
      const int64_t &x,
      const int64_t &y,
      std::vector<std::vector<int>> &simulation_space
      ) {
    std::random_device rd;
    std::mt19937 g(rd());
    int64_t new_x = x;
    int64_t new_y = y;

    while (new_x >= 0 && new_x < this->SIMULATION_SIZE && new_y >= 0 && new_y < this->SIMULATION_SIZE) {
      if (this->collision(new_x, new_y, simulation_space)) {
        std::cout << "Writing to simulation space!" << std::endl;
        write_simulation_space(new_x, new_y, simulation_space, 1);
        return;
      }

      new_x += g() % 2;
      new_y += g() % 2;
    }
  }

  bool Crystal::collision(int64_t x, int64_t y, const std::vector<std::vector<int>>& simulation_space) {
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        const int64_t t_x = x + i;
        const int64_t t_y = y + j;

        if (t_x >= 0 && t_x < this->SIMULATION_SIZE &&
            t_y >= 0 && t_y < this->SIMULATION_SIZE &&
            read_simulation_space(t_x, t_y, simulation_space) == 1) {
          if (read_simulation_space(t_x, t_y, simulation_space) == 1) {
            return true;
          }
        }
      }
    }

    return false;
  }

  std::tuple<int64_t, int64_t> Crystal::insert_particle(
      const std::vector<std::vector<int>>& simulation_space,
      const int radius
      ) {
    std::random_device rd;
    std::mt19937 g(rd());
    long int random_row = 0;
    long int random_col = 0;

    do {
      random_row = g() % this->ROWS;
      random_col = g() % this->COLS;
    } while ((abs(this->CENTER - random_row) <= radius + 1 &&
              abs(this->CENTER - random_col) <= radius - 1) ||
              read_simulation_space(random_row, random_col, simulation_space) != 0);

    return std::make_tuple(random_row, random_col);
  }

}// namespace crystal
