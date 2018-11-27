#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>

template <unsigned n>
class NQueens {
  public:
    /**
     * Prints the game board
     */
    void print(const std::vector<std::vector<int>> game_board) {
      for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < n; ++j) {
          std::cout << game_board[i][j] << " " << std::flush;
        }
        std::cout << std::endl;
      }
    }


    /**
     * Sequential runs until all n queens
     * have been placed
     */
    auto sequential() {
      std::vector<std::vector<std::string>> valid;
      std::vector<int> queens;
      sequential_helper(valid, queens);

      return valid.size();
    }

    void sequential_helper(std::vector<std::vector<std::string>>& done, std::vector<int>& queens) {
      if (queens.size() == n) {
        done.emplace_back();
        for (auto i = 0u; i < n; ++i) {
          done.back().emplace_back(n, '.');
          done.back().back()[queens[i]] = 'Q';
        }
      } else {
        for (auto i = 0u; i < n; ++i) {
          bool valid = true;

          for (auto j = 0u; valid && j < queens.size(); ++j) {
            valid = (queens[j] != i) && (abs(queens[j] - i) != queens.size() - j);
          }

          if (valid) {
            queens.push_back(i);
            sequential_helper(done, queens);
            queens.pop_back();
          }
        }
      }
    }

    /**
     * Parallel runs in MPI until all
     * queens have been placed
     */
    void parallel() {
      MPI_Request req;
      int rank, num_nodes;

      MPI_Init(nullptr, nullptr);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

      std::shared_ptr<int> return_values(new int[2 * n]);
      int data_size, machine = 0;

      if (rank == this->MASTER) {

      } else {
        for (;;) {
          MPI_Isend(&machine, 1, MPI_INT, this->MASTER, this->TAG, MPI_COMM_WORLD, &req);
          MPI_Wait(&req, MPI_STATUS_IGNORE);

          MPI_Recv(&data_size, 1, MPI_INT, this->MASTER, this->TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          if (data_size < 0) {
            break;
          }

          MPI_Recv(reinterpret_cast<void*>(&return_values), data_size * 2, MPI_INT, this->MASTER, this->TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          this->sequential();
        }
      }

      MPI_Finalize();
    }

  private:
    const int MASTER = 0;
    const int TAG = 0;
};

int main(int argc, char** argv) {
  NQueens<11> nq;
  auto value = nq.sequential();
  std::cout << value << std::endl;

  return EXIT_SUCCESS;
}
