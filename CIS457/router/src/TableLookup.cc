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
						if (column.find("/") > 0) {
							column = column.substr(0, column.find("/"));
						}
            columns.push_back(column);
        }

        this->prefix_interface_table.insert(std::pair<std::string, std::string>(columns[0], columns[2]));
        std::cout << "Adding " << columns[0] << " (" << columns[2]  << ") to network table..." << std::endl;
        if (columns[1] != "-") {
            this->hop_device_table.insert(std::pair<std::string, std::string>(columns[0], columns[1]));
						std::cout << "Adding " << columns[0] << " - " << columns[1] << " to hop table..." << std::endl;
        }

        columns.clear();
      }
		tableFile.close();
  }

  std::string TableLookup::get_route(std::string route) {
		std::string route_1 = route.substr(0, 7) + "0";
		std::string route_2 = route.substr(0, 5) + "0.0";
		std::unordered_map<std::string, std::string>::const_iterator index_1 = prefix_interface_table.find(route_1);
		if (index_1 != prefix_interface_table.end()) {
			return index_1->second;
		}
		std::unordered_map<std::string, std::string>::const_iterator index_2 =
prefix_interface_table.find(route_2);
		if (index_2 != prefix_interface_table.end()) {
			return index_2->second;				
		}
		return "";
  }

	std::string TableLookup::get_hop_device(std::string route) {
		std::string route_1 = route.substr(0, 7) + "0";
		std::string route_2 = route.substr(0, 5) + "0.0";
		std::unordered_map<std::string, std::string>::const_iterator index_1 = hop_device_table.find(route_1);
		if (index_1 != hop_device_table.end()) {
			return index_1->second;
		}
		std::unordered_map<std::string, std::string>::const_iterator index_2 =
hop_device_table.find(route_2);
		if (index_2 != hop_device_table.end()) {
			return index_2->second;				
		}
		return "";
  }
} // namespace router
