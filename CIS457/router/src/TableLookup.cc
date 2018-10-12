#include "../include/router/TableLookup.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace router {
    TableLookup::TableLookup(const std::string& filename) {
        std::ifstream tableFile(filename);
        std::string line;
        if (tableFile.is_open()) {
            while (std::getline(tableFile, line)) {
                std::vector<std::string> columns;
                std::stringstream stream(line);
                std::string column;
                while (std::getline(stream, column, ' ')) {
                    columns.push_back(column);
                }

                this->prefixInterfaceTable.insert(std::pair<std::string, std::string>(columns[0], columns[2]));

                if (columns[1] != "-") {
                    this->hopDeviceTable.insert(std::pair<std::string, std::string>(columns[0], columns[1]));
                }

                columns.clear();
            }
        }

        tableFile.close();
    }

    std::string TableLookup::getRoute(const std::string& route) {
        return this->prefixInterfaceTable.find(route)->second;
    }

    bool TableLookup::hasHopDevice(const std::string& route) {
        auto it = this->hopDeviceTable.begin();
        return it != this->hopDeviceTable.end();
    }
}