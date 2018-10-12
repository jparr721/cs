#ifndef INCLUDE_ROUTER_TABLE_LOOKUP_HPP
#define INCLUDE_ROUTER_TABLE_LOOKUP_HPP

#include <string>
#include <unordered_map>

namespace router {
    class TableLookup {
    public:
        explicit TableLookup(const std::string&);
        std::string getRoute(const std::string&);
        bool hasHopDevice(const std::string&);
    private:
        const int MAX_COLUMNS = 3;
        std::unordered_map<std::string, std::string> prefixInterfaceTable;
        std::unordered_map<std::string, std::string> hopDeviceTable;
    };
}

#endif
