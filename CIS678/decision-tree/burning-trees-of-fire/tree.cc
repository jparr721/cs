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
    Node root;
    root.index = 0;
    tree.push_back(root);
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
      return;
    }

    int selected_idx = select_max_gain(table);
    std::map<std::string, std::vector<int>> attr_map;

    for (size_t i = 0u; i < table.data.size(); ++i) {
      attr_map[table.data[i][selected_idx]].push_back(i);
    }

    tree[index].index = selected_idx;
    auto majority_pair = get_majority_class_label(table);
    std::cout << majority_pair.first << " --- " << majority_pair.second << std::endl;
    double total_proportion = (double)majority_pair.second / table.data.size();

    // Assume it is a mostly pure sample in this case
    // If it's a leaf, we can just blast this answer
    if (total_proportion > 0.8) {
      tree[index].is_leaf = true;
      tree[index].label = majority_pair.first;
      return;
    }

    // If it's not a majority label, we need to make one
    for (size_t i = 0u; i < initial_table.data_value_list[selected_idx].size(); ++i) {
      std::string value = initial_table.data_value_list[selected_idx][i];

      Table new_table;
      std::vector<int> attr_indexes = attr_map[value];
      for (size_t i = 0; i < attr_indexes.size(); ++i) {
        new_table.data.push_back(table.data[attr_indexes[i]]);
      }

      Node next_node;
      next_node.value = value;

      // Since we always add to the bottom, make it current tree size
      next_node.tree_index = (int)tree.size();

      // Stack another child node location onto the tree
      tree[index].children.push_back(next_node.tree_index);

      // Push back the next node
      tree.push_back(next_node);

      // If the table data is empty
      if (new_table.data.size() == 0) {
        next_node.is_leaf = true;
        next_node.label = get_majority_class_label(new_table).first;
        tree[next_node.index] = next_node;
      } else {
        // If not empty, recurse down the subtree
        std::cout << new_table.data.size() << std::endl;
        std::cout << new_table.attribute_list.size() << std::endl;
        std::cout << next_node.label << std::endl;
        fit(new_table, next_node.index);
      }
    }
  }

  void Tree::print_tree(int idx, std::string branch) {
    if (tree[idx].is_leaf) {
      std::cout << branch << "Label: " << tree[idx].label << std::endl;
    }

    for (size_t i = 0; i < tree[idx].children.size(); ++i) {
      int child_idx = tree[idx].children[i];
      std::string attr_name = initial_table.attribute_list[tree[idx].index];
      std::string attr_value = tree[child_idx].value;

      print_tree(child_idx, branch + attr_name + " = " + attr_value + ", ");
    }
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
  std::cout << "The data" << std::endl;
  tree.print_mat<std::string>(tree.initial_table.data);
  tree.print_mat<std::string>(tree.initial_table.data_value_list);
  std::cout << "attributes..." << std::endl;
  for (const auto& value : tree.initial_table.attribute_list) {
    std::cout << value << std::endl;
  }

  tree.fit(tree.initial_table, 0);
  std::cout << "Tree generated..." << std::endl;
  tree.print_tree(0, "");

  return EXIT_SUCCESS;
}
