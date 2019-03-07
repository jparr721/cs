#include <cmath>
#include <map>
#include <numeric>
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
    initial_table.init(*data_);
  }

  void Tree::calculate_total_entropy() {
    const auto dataset = data_->attribute_values;

    std::unordered_map<std::string, double> occurances;

    // Sum the classes
    for (const auto& row : dataset) {
      ++occurances[row[row.size() - 1]];
    }

    const auto vals = util::extract_values<std::unordered_map<std::string, double>, double>(occurances);
    const auto total = std::accumulate(vals.begin(), vals.end(), 0.0);

    double final_entropy = (occurances.begin()->second/total) * std::log2(occurances.begin()->second/total);
    for (auto it = std::next(occurances.begin(), 1); it != occurances.end(); ++it) {
      final_entropy -= (it->second/total * std::log2(it->second / total));
    }

    total_entropy_ = final_entropy;
  }

  void Tree::fit(const Table& table, int index) {
    if (is_leaf_node(table)) {
      tree[index].is_leaf = true;
      tree[index].label = table.data.back().back();
    }
  }

  double Tree::attribute_entorpy(const Table& table, int index) {
    return 0.0;
  }

  double Tree::gain(const Table& table, int index) {
    return total_entropy_ - attribute_entorpy(table, index);
  }

  int Tree::select_max_gain(const Table& table) {
    int idx{-1};
    int max_gain{0.0};

    for (size_t i = 0; i < initial_table.size(); ++i) {
      if (max_gain < gain(Table, i)) {
        std::cout << "Fuck off" << std::endl;
      }
    }
  }

  std::string Tree::choose(const std::vector<std::string>& row) {
    // Recurse until we know it's a leaf node
    int leaf = dfs(row, 0);
    return leaf != -1 ? tree[leaf].label : "fail";
  }

  bool Tree::is_leaf_node(const Table& table) {
    for (size_t i = 1u; i < table.data.size(); ++i) {
      if (table.data[0].back() != table.data[i].back()) {
        return false;
      }
    }

    return true;
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
