#ifndef TREE_H_
#define TREE_H_

#include <memory>
#include <utility>
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
      void print_tree(int idx, std::string branch);

      template <typename T>
      void print_mat(const std::vector<std::vector<T>> mat) {
        for (const auto& row : mat) {
          for (const auto& col : row) {
            std::cout << col << ",";
          }
          std::cout << std::endl;
        }
      }

      Table initial_table;

    private:
      void calculate_total_entropy();

      int dfs(const std::vector<std::string>& row, int current);
      int select_max_gain(const Table& table);

      double gain(const Table& table, int index) const;
      double attribute_entropy(const Table& table, int index) const;
      double single_attribute_entropy(const Table& table) const;

      bool is_leaf_node(const Table& table);

      std::string choose(const std::vector<std::string>& row);

      std::pair<std::string, int> get_majority_class_label(Table table);

      double total_entropy_;
      std::unique_ptr<dataset> data_;
      std::vector<Node> tree;
  };
} // namespace tree

#endif // TREE_H_
