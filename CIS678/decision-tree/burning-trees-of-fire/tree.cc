#include <map>
#include "tree.h"
#include "util.h"

namespace tree {
  void Table::init(dataset the_data) {
    attribute_list = the_data.targets;
    data = the_data.attribute_values;
    data_value_list.resize(attribute_list.size());

    for (size_t i = 0u; i < attribute_list.size(); ++i) {
      std::map<std::string, int> value;

      for (const auto& val : data) {
        value[val[i]] = 1;
      }

      for (const auto& map_val : value) {
        data_value_list[i].push_back(map_val.first);
      }
    }
  }

  Tree::Tree(std::unique_ptr<dataset> input) {
    data_ = std::move(input);
    Table initial_table;
    initial_table.init(*data_);
  }

  std::string Tree::choose(const std::vector<std::string>& row) {
    // Recurse until we know it's a leaf node
    int leaf = dfs(row, 0);
    return leaf != -1 ? tree[leaf].label : "fail";
  }

  int Tree::dfs(const std::vector<std::string>& row, int index) {
    if (tree[index].is_leaf) {
      return index;
    }

    int t_index = tree[index].index;

    for (size_t i = 0u; i < tree[index].children.size(); ++i) {
      int next_index = tree[index].children[i];

      // If not a leaf, keep going
      if (row[t_index] == tree[next_index].value) {
        dfs(row, next_index);
      }
    }

    return -1;
  }
} // namespace tree
