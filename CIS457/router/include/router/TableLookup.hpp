#ifndef INCLUDE_ROUTER_TABLE_LOOKUP_HPP
#define INCLUDE_ROUTER_TABLE_LOOKUP_HPP

#define MAX_COLUMNS 3

namespace router {
    class TableLookup {
    public:
        TableLookup(const std::string&) = default;
        std::string getRoute(const std::string&);
        bool hasHopDevice(cosnt std::string&);
    private:
        std::map<string, string> prefixInterfaceTable;
        std::map<string, string> hopDeviceTable;
    };
}

#endif