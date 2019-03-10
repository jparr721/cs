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

    int selected_idx = select_max_gain(table);
    std::map<std::string, std::vector<int>> attr_map;

    for (size_t i = 0u; i < table.data.size(); ++i) {
      attr_map[table.data[i][selected_idx]].push_back(i);
    }

    tree[index].index = selected_idx;
    auto majority_pair = get_majority_class_label(table);
    double total_proportion = (double)majority_pair.second / table.data.size();

    // Assume it is a mostly pure sample in this case
    // If it's a leaf, we can just blast this answer
    if (total_proportion > 0.8) {
      tree[index].is_leaf = true;
      tree[index].label = majority_pair.first;
      return;
    }

    // If it's not a majority label, we need to make one
  }

  std::pair<std::string, int>  Tree::get_majority_class_label(Table table) {
    std::string label("");
    int count{0};

    std::map<std::string, int> counts;

    for (size_t i = 0; i < table.data.size(); ++i) {
      counts[table.data[i].back()]++;

      if (counts[table.data[i].back()] > count) {
        count = counts[table.data[i].back()];
        label = table.data[i].back();
      }
    }

    return {label, count};
  }

  double Tree::single_attribute_entropy(const Table& table) const {
    double ret{0.0};
    int total = (int) table.data.size();

    std::map<std::string, int> counts;

    for (size_t i = 0; i < table.data.size(); ++i) {
      counts[table.data[i].back()]++;
    }

    for (const auto& count : counts) {
      double p = (double)count.second / total;

      ret += -1.0 * p * std::log2(p);
    }
    return ret;
  }

  double Tree::attribute_entropy(const Table& table, int index) const {
    double ret{0.0};
    int total = (int)table.data.size();

    std::map<std::string, std::vector<int>> attr_map;
    for (size_t i = 0u; i < table.data.size(); ++i) {
      attr_map[table.data[i][index]].push_back(i);
    }

    for (const auto& val : attr_map) {
      Table new_table;
      for (size_t i = 0u; i < val.second.size(); ++i) {
        new_table.data.push_back(table.data[val.second[i]]);
      }

      int next_item_count = (int) new_table.data.size();

      ret += (double) next_item_count / total * single_attribute_entropy(new_table);
    }

    return ret;
  }

  double Tree::gain(const Table& table, int index) const {
    return total_entropy_ - attribute_entropy(table, index);
  }

  int Tree::select_max_gain(const Table& table) {
    int idx{-1};
    double max_gain{0.0};

    for (size_t i = 0; i < initial_table.data_value_list.size(); ++i) {
      auto gain_ratio = gain(table, i);
      if (max_gain < gain_ratio) {
        max_gain = gain_ratio;
        idx = i;
      }
    }

    return idx;
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

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: burn <input_path>" << std::endl;
    return EXIT_FAILURE;
  }

  tree::Loader loader;
  auto data = loader.load(argv[1]);

  tree::Tree tree(std::move(data));
  tree.print_mat<std::string>(tree.initial_table.data);

  return EXIT_SUCCESS;
}
