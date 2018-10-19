#ifndef INCLUDE_ROUTER_ROUTER_HPP
#define INCLUDE_ROUTER_ROUTER_HPP

#include <vector>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/ether.h>
#include "ARPHeader.hpp"
#include <string>
#include <sys/types.h>

namespace router {
class Router {
  public:
    Router() = default;
    ~Router() = default;

    ARPHeader* build_arp_reply(
      struct ether_header *eh,
      struct ether_arp *arp_frame,
      unsigned char destination_mac[6]);
    uint16_t checksum(unsigned char* addr, int len);
		std::string get_ip_str(unsigned char[4]);
    int Start();
};
} // namespace router

#endif
