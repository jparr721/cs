#ifndef INCLUDE_ROUTER_NETWORK_INTERFACE_HPP
#define INCLUDE_ROUTER_NETWORK_INTERFACE_HPP

#include <string>

namespace router {
class NetworkInterface {
  public:
    char* name;
    int descr;
    unsigned char mac_addr[6];
    unsigned char ip_addr[4];
};
} // namespace router

#endif
