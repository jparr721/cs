#ifndef TREE_H_
#define TREE_H_

#include <memory>
#include <vector>
#include "loader.h"
#include "node.h"

namespace tree {
  class Table {
    std::string attr;
    std::vector<std::vector<std::string>> data;
    std::vector<std::vector<std::string>> data_value_list;
  };

  class Tree {
    public:
      Tree(std::unique_ptr<dataset> input) : data_(std::move(input)) {};

      int dfs(const std::vector<std::string>& row, int current);

      void fit(Table t, int index);

    private:
      std::unique_ptr<dataset> data_;
      std::vector<Node> tree;
  };
} // namespace tree

#endif // TREE_H_
