#include <algorithm>
#include <chrono>
/* #include <crystal/crystal.hpp> */
#include <fstream>
#include "../include/crystal/crystal.hpp"
#include <iostream>
#include <iterator>
#include <random>
#include "omp.h"

namespace crystal {
  auto read_simulation_space = [](const long int x, const long int y, const std::vector<std::vector<int>>& simulation_space)->int {
    #pragma omp atomic read
    auto val = simulation_space[x][y];
    return val;
  };

  auto write_simulation_space = [](const long int x, const long int y, std::vector<std::vector<int>>& simulation_space, int val)->void {
    #pragma omp atomic write
    simulation_space[x][y] = val;
  };

  auto read_radius = [](const int& radius)->int {
    #pragma omp atomic read
    auto val = radius;
    return val;
  };

  auto write_radius = [](int& radius, int val)->void {
    #pragma omp atomic write
    radius = val;
  };

  Crystal::Crystal(long int particles, long int simulation_size) {
    this->SIMULATION_SIZE = simulation_size;
    this->ROWS = simulation_size;
    this->COLS = simulation_size;
    this->MAX_MOVES = 4 * simulation_size;
    this->CENTER = simulation_size / 2;
  }

  void Crystal::Run(long int particles) {
    int radius = 0;
    std::vector<std::vector<int>> simulation_space(
        this->ROWS,
        std::vector<int>(this->COLS));

    // Place our point in the center of the board
    simulation_space[this->CENTER][this->CENTER] = 1;

    #pragma omp parallel
    {
      for (int i = 0; i < particles; ++i) {
        if (read_radius(radius) >= this->SIMULATION_SIZE / 2) {
          break;
        }

        const auto point_location = this->insert_particle(simulation_space, radius);
        long int x = std::get<0>(point_location);
        long int y = std::get<1>(point_location);
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
  }

  /**
   * Takes the simulation space and
   * maps it into a file for processing
   * in the python interpreter
   */
  void Crystal::end_simulation(const std::vector<std::vector<int>> &simulation_space) {
    std::ofstream the_goods;

    the_goods.open("output.txt");

    for (long unsigned int i = 0; i < simulation_space.size(); ++i) {
      for (long unsigned int j = 0; j < simulation_space[i].size(); ++j) {
        int current = 0;

        if (simulation_space[i][j] == 1) {
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
    for (long unsigned int i = 0; i < this->SIMULATION_SIZE; ++i) {
      for (long unsigned int j = 0; j < this->SIMULATION_SIZE; ++j) {
        int current = simulation_space[i][j];

        std::cout << current << ", " << std::endl;
      }

      std::cout << "\n" << std::endl;
    }
  }

  void Crystal::random_walk(
      long int &x,
      long int &y,
      std::vector<std::vector<int>> &simulation_space
      ) {
    std::random_device rd;
    std::mt19937 g(rd());

    while (x >= 0 && x < this->SIMULATION_SIZE && y >= 0 && y < this->SIMULATION_SIZE) {
      if (this->collision(x, y, simulation_space)) {
        std::cout << "Writing to simulation space!" << std::endl;
        write_simulation_space(x, y, simulation_space, 1);
        return;
      }

      x += g() % 2;
      y += g() % 2;
    }
  }

  bool Crystal::collision(long int x, long int y, const std::vector<std::vector<int>>& simulation_space) {
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        const long int t_x = x + i;
        const long int t_y = y + j;

        if (t_x >= 0 && t_x < this->SIMULATION_SIZE && t_y >= 0 && t_y < this->SIMULATION_SIZE && read_simulation_space(t_x, t_y, simulation_space) == 1) {
          if (read_simulation_space(t_x, t_y, simulation_space) == 1) {
            return true;
          }
        }
      }
    }

    return false;
  }

  std::tuple<long int, long int> Crystal::insert_particle(const std::vector<std::vector<int>>& simulation_space, const int radius) {
    std::random_device rd;
    std::mt19937 g(rd());
    long int random_row = 0;
    long int random_col = 0;

    do {
      random_row = g() % this->ROWS;
      random_col = g() % this->COLS;
    } while ((abs(this->CENTER - random_row) <= radius + 1 && abs(this->CENTER - random_col) <= radius - 1) || read_simulation_space(random_row, random_col, simulation_space) != 0);

    return std::make_tuple(random_row, random_col);
  }

}// namespace crystal
