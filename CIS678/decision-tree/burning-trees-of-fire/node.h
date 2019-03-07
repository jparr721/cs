#ifndef NODE_H_
#define NODE_H_

#include <string>
#include <vector>

namespace tree {
  struct Node {
    int index;
    int tree_index;

    bool is_leaf = false;

    std::string value;
    std::string label;

    // Integers to keep track of chil index
    std::vector<int> children;
  };
} // namespace tree

#endif // NODE_H_
