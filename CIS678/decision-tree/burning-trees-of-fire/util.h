#ifndef UTIL_H_
#define UTIL_H_

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace util {
  template<typename MapType, typename TK>
  inline std::vector<TK> extract_keys(const MapType& input_map) {
    std::vector<TK> retval;
    for (const auto& element : input_map) {
      retval.push_back(element.first);
    }
    return retval;
  }

  template<typename MapType, typename TV>
  inline std::vector<TV> extract_values(const MapType& input_map) {
    std::vector<TV> retval;
    for (const auto& element : input_map) {
      retval.push_back(element.second);
    }
    return retval;
  }

  template <typename T>
  inline std::string vec_to_string(const std::vector<T>& values) {
    std::string retval;
    for (const auto& val : values) {
      retval += val + ",";
    }

    return retval;
  }

  template <typename T>
  inline void print_2d_vector(const std::vector<std::vector<T>>& values) {
    for (const auto& val : values) {
      for (const auto& v : val) {
        std::cout << v << ",";
      }
      std::cout << std::endl;
    }
  }

  inline bool is_integer(const std::string & s) {
     if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;

     char * p;
     strtol(s.c_str(), &p, 10);

     return (*p == 0);
  }

  inline std::vector<std::string> split(const std::string& line, char delim) {
    std::vector<std::string> result;
    std::istringstream iss(line);
    std::string the_line = "";

    while (std::getline(iss, the_line, delim))
      result.push_back(the_line);

    return result;
  }
} // namespace util

#endif // UTIL_H_
