/* #include <router/TableLookup.hpp> */
#include "../include/router/TableLookup.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace router {
  TableLookup::TableLookup(const std::string& filename) {
		std::cout << "Loading network table..." << std::endl;
    std::ifstream tableFile(filename);
    std::string line;
    
			while (!tableFile.eof()) {
				getline(tableFile, line);
        std::vector<std::string> columns;
        std::stringstream stream(line);
        std::string column;

        if (line.length() <= 0) {
						break;
				}

        while (std::getline(stream, column, ' ')) {
            columns.push_back(column);
        }

        this->prefix_interface_table.insert(std::pair<std::string, std::string>(columns[0], columns[2]));
        std::cout << "Adding " << columns[0] << " to network table..." << std::endl;
        if (columns[1] != "-") {
            this->hop_device_table.insert(std::pair<std::string, std::string>(columns[0], columns[1]));
						std::cout << "Adding " << columns[0] << " - " << columns[1] << " to hop table..." << std::endl;
        }

        columns.clear();
      }
		tableFile.close();
  }

  std::string TableLookup::get_route(const std::string& route) {
    return this->prefix_interface_table.find(route)->second;
  }

  bool TableLookup::has_hop_device(const std::string& route) {
    auto it = this->hop_device_table.find(route);
    return it != this->hop_device_table.end();
  }
} // namespace router
