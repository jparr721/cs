#ifndef LOADER_H_
#define LOADER_H_

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "util.h"

namespace tree {
  struct dataset {
    dataset(
      int nt,
      std::vector<std::string> t,
      int na,
      std::map<std::string, std::vector<std::string>> a,
      int ne,
      std::vector<std::vector<std::string>> av) :
    num_targets(nt), targets(t), num_attributes(na), attributes(a), num_examples(ne), attribute_values(av) {};

    int num_targets;
    std::vector<std::string> targets;

    int num_attributes;
    std::map<std::string, std::vector<std::string>> attributes;

    int num_examples;
    std::vector<std::vector<std::string>> attribute_values;

  };

  class Loader {
    public:
      inline std::unique_ptr<dataset> load(const std::string& path) {
        std::ifstream indata;
        indata.open(path);

        std::string line;
        std::vector<std::string> lines;

        // Num targets, num attr, num examples
        int nt, na, ne = -1;

        // Get all that stuff loaded
        while (std::getline(indata, line)) {
          std::stringstream ss(line);
          lines.push_back(ss.str());
        }

        nt = std::stoi(lines[0]);
        na = std::stoi(lines[2]);
        ne = std::stoi(lines[2 + na + 1]);

        auto targets = util::split(lines[1], ',');
        std::map<std::string, std::vector<std::string>> attributes;

        for (int i = 0; i < na; ++i) {
          int idx = i + 4;
          auto vals = util::split(lines[idx], ',');

          std::vector<std::string> sub(vals.begin() + 2, vals.end());
          attributes[vals[0]] = sub;
        }

        std::vector<std::vector<std::string>> attribute_values;
        for (uint i = ne + 1; i < lines.size(); ++i) {
          auto vals = util::split(lines[i], ',');
          for (const auto& val : vals) {
            std::vector<std::string> ov;
            ov.push_back(val);
            attribute_values.push_back(ov);
          }
        }

        // Return a unique ptr to our shared data
        return std::make_unique<dataset>(nt, targets, na, attributes, ne, attribute_values);
      }
  };
} // namespace tree

#endif // LOADER_H_
