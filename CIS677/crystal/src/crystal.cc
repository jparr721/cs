#include <algorithm>
#include <chrono>
/* #include <crystal/crystal.hpp> */
#include <fstream>
#include "../include/crystal/crystal.hpp"
#include <iostream>
#include <iterator>
#include <random>

namespace crystal {
  Crystal::Crystal(int particles, int simulation_size) {
    this->SIMULATION_SIZE = simulation_size;
    this->ROWS = simulation_size;
    this->COLS = simulation_size;
    this->MAX_MOVES = 4 * simulation_size;
    this->CENTER = simulation_size / 2;
  }

  void Crystal::Run(int particles) {
    int radius = 0;
    std::vector<std::vector<int>> simulation_space(
        this->ROWS,
        std::vector<int>(this->COLS));

    // Place our point in the center of the board
    simulation_space[this->CENTER][this->CENTER] = 1;

    for (int i = 0; i < particles; ++i) {
      if (radius >= this->SIMULATION_SIZE / 2) {
        break;
      }

      const auto point_location = this->insert_particle(radius);
      std::cout << "particle created, walking" << std::endl;
      int x = std::get<0>(point_location);
      int y = std::get<1>(point_location);
      this->random_walk(x, y, simulation_space);

      if (x >= 0 && x < this->SIMULATION_SIZE && y >= 0 && y < this->SIMULATION_SIZE) {
        const int distance = std::max(std::abs(this->CENTER - x), std::abs(this->CENTER - y));

        if (distance > radius) {
          radius = distance;
        }
      }
    }

    this->end_simulation(simulation_space);
  }

  /**
   * Takes the simulation space and
   * maps it into a file for processing
   * in the python interpreter
   */
  void Crystal::end_simulation(const std::vector<std::vector<int>> &simulation_space) {
    std::ofstream the_goods;

    the_goods.open("output.txt");

    for (int i = 0; i < simulation_space.size(); i++) {
      for (int j = 0; j < simulation_space[i].size(); j++) {
        int current = 0;

        if (simulation_space[i][j] == 1) {
          current = 1;
        }

        if (j != 0) {
          the_goods << ", ";
        }

        the_goods << current;
      }
      the_goods << "\n";
    }

    the_goods.close();
  }

  void Crystal::print(const std::vector<std::vector<int>>& simulation_space) {
    for (int i = 0; i < this->SIMULATION_SIZE; i++) {
      for (int j = 0; j < this->SIMULATION_SIZE; j++) {
        int current = simulation_space[i][j];

        std::cout << current << ", " << std::endl;
      }

      std::cout << "\n" << std::endl;
    }
  }

  void Crystal::random_walk(
      int &x,
      int &y,
      std::vector<std::vector<int>> &simulation_space
      ) {
    std::random_device rd;
    std::mt19937 g(rd());

    while (x >= 0 && x < this->SIMULATION_SIZE && y >= 0 && y < this->SIMULATION_SIZE) {
      std::cout << x << std::endl;
      std::cout << y << std::endl;
      if (this->collision(x, y, simulation_space)) {
        simulation_space[x][y] = 1;
        return;
      }

      x += g() % 2;
      y += g() % 2;
    }
  }

  bool Crystal::collision(int x, int y, const std::vector<std::vector<int>>& simulation_space) {
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        int t_x = x + i;
        int t_y = y + j;
        std::cout << "t_x: " << t_x << std::endl;
        std::cout << "t_y: " << t_y << std::endl;

        if (t_x >= 0 && t_x < this->SIMULATION_SIZE && t_y >= 0 && t_y < this->SIMULATION_SIZE) {
          if (simulation_space[t_x][t_y] == 1) {
            return true;
          }
        }
      }
    }

    return false;
  }

  bool Crystal::valid_coordinates(const std::vector<std::vector<int>>& simulation_space, int x, int y) {
    return x < this->SIMULATION_SIZE && y < this->SIMULATION_SIZE;
  }

  std::tuple<int, int> Crystal::insert_particle(const int radius) {
    std::random_device rd;
    std::mt19937 g(rd());
    int random_row = 0;
    int random_col = 0;

    do {
      random_row = g() % this->ROWS;
      random_col = g() % this->COLS;
    } while (abs(this->CENTER - random_row) <= radius + 1 && abs(this->CENTER - random_col) <= radius - 1);

    return std::make_tuple(random_row, random_col);
  }

}// namespace crystal
