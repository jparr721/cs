#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>

class NQueens {
  public:
    explicit NQueens(int n) : n(n) {}
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
    void parallel(int argc, char** argv) {
      MPI_Request req;
      int rank, num_nodes;

      MPI_Init(&argc, &argv);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

      long valid_solutions = sequential();
      long valid;

      if (rank != this->MASTER) {
        MPI_Send(&valid_solutions, 1, MPI_INT, this->MASTER, this->TAG, MPI_COMM_WORLD);
      } else {
        valid = valid_solutions;
        for (int i = 1; i < num_nodes; ++i) {
          MPI_Recv(&valid, 1, MPI_INT, i, this->TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::cout << valid << std::endl;
      }

      MPI_Finalize();
    }

  private:
    const int MASTER = 0;
    const int TAG = 0;
    uint16_t n;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: nq n" << std::endl;
    return EXIT_FAILURE;
  }
  int n = std::stoi(argv[1]);

  NQueens nq(n);
  /* auto value = nq.sequential(); */
  nq.parallel(argc, argv);

  return EXIT_SUCCESS;
}
