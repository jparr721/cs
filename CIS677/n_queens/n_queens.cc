#include <cstdint>
#include <iostream>
#include <vector>

template <unsigned n>
class NQueens {
  public:
    /**
     * Prints the game board
     */
    void print(const std::vector<std::vector<int>> game_board) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
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
      if constexpr (n == 0) {
        return false;
      }
      // Start with 2d vector of zeros with game board dims
      std::vector<std::vector<int>> game_board(n, std::vector<int>(n, 0));
      if (!sequential_helper(game_board, 0)) {
        std::cout << "No solutions." << std::endl;
        return EXIT_SUCCESS;
      }

      print(game_board);
      return EXIT_SUCCESS;
    }

    bool valid_spot(const std::vector<std::vector<int>>& game_board, int row, int col) const {
      for (int i = 0; i < col; ++i) {
        if (game_board[row][i])
          return false;

        for (int i = row, j = col; i >= 0 && j >=0; --i, --j) {
          if (game_board[i][j])
            return false;
        }

        for (int i = row, j = col; j >= 0 && i < n; ++i, --j) {
          if (game_board[i][j])
            return false;
        }
      }
      return true;
    }

    bool sequential_helper(std::vector<std::vector<int>>& game_board, int col) {
      if (col >= n) {
        return true;
      }

      for (int i = 0; i < n; ++i) {
        if (valid_spot(game_board, i, col)) {
          game_board[i][col] = 1;

          // Recurse!
          if (sequential_helper(game_board, col + 1))
            return true;

          // Backtrack if this doesn't work
          game_board[i][col] = 0;
        }
      }

      return false;
    }

    /**
     * Parallel runs in MPI until all
     * queens have been placed
     */
    void parallel() {
      // TODO
    }
};

int main() {
  NQueens<8> nq;
  nq.sequential();

  return EXIT_SUCCESS;
}
