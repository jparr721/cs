#ifndef TREE_H_
#define TREE_H_

#include <memory>
#include <vector>
#include "loader.h"
#include "node.h"

namespace tree {
  class Table {
    public:
      std::vector<std::string> attribute_list;
      std::vector<std::vector<std::string>> data;
      std::vector<std::vector<std::string>> data_value_list;

      void init(dataset the_data);
  };

  class Tree {
    public:
      Tree(std::unique_ptr<dataset> input);
      int dfs(const std::vector<std::string>& row, int current);
      std::string choose(const std::vector<std::string>& row);

      void fit(Table t, int index);

    private:
      std::unique_ptr<dataset> data_;
      std::vector<Node> tree;
  };
} // namespace tree

#endif // TREE_H_
