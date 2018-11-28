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

    void parallel2(int argc, char** argv) {
      int num_nodes, rank, buf, sol = 0;
      MPI_Init(&argc, &argv);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

      if (num_nodes < n) {
        for (auto i = 0u; i < n; ++i) {
          auto solutions = sequential();
          sol += solutions;
        }
      } else {
        sol = sequential();
      }

      if (rank == MASTER) {
        for (int i = 1; i < num_nodes; ++i) {
          MPI_Recv(&buf, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          sol += buf;
        }

        std::cout << "Solutions: " << sol << std::endl;
      } else {
        MPI_Send(&sol, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD);
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

      int data_size;
      std::unique_ptr<int[]> return_values(new int[2 * n]);
      std::vector<int> queens;
      std::vector<std::vector<std::string>> done;

      if (rank == this->MASTER) {
        std::unique_ptr<int[]> solutions(new int[2 * n]);
        std::unique_ptr<MPI_Request[]> requests(new MPI_Request[num_nodes - 1]);
        for (int i = 0; i < num_nodes; ++i) {
          MPI_Irecv(&solutions[i], 1, MPI_INT, i, TAG, MPI_COMM_WORLD, &requests[i]);
        }
      } else {
        for (int i = 0; i < num_nodes; ++i) {
          MPI_Isend(&i, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, &req);
          MPI_Wait(&req, MPI_STATUS_IGNORE);
          MPI_Recv(&data_size, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          if (data_size <= 0) {
            std::cerr << "Failed to hit node: " << i << std::endl;
            break;
          }

          MPI_Recv(reinterpret_cast<void*>(&return_values), data_size * 2, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
          return_values[i] = done.size();
        }
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
  auto value = nq.sequential();
  std::cout << value << std::endl;
  /* nq.parallel(argc, argv); */
  /* nq.parallel2(argc, argv */

  return EXIT_SUCCESS;
}
