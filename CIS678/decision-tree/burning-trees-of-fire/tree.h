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

      void fit(const Table& table, int index);

    private:
      void calculate_total_entropy();

      int dfs(const std::vector<std::string>& row, int current);
      int select_max_gain(const Table& table);

      double gain(const Table& table, int index);
      double attribute_entropy(const Table& table, int index);

      bool is_leaf_node(const Table& table);

      std::string choose(const std::vector<std::string>& row);

      double total_entropy_;
      Table initial_table;
      std::unique_ptr<dataset> data_;
      std::vector<Node> tree;
  };
} // namespace tree

#endif // TREE_H_
