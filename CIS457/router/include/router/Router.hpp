#ifndef INCLUDE_ROUTER_ROUTER_HPP
#define INCLUDE_ROUTER_ROUTER_HPP

#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/ether.h>
#include "ARPHeader.hpp"
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace router {
class Router {
  public:
    Router() = default;
    ~Router() = default;

    ARPHeader* build_arp_reply(
      struct ether_header *eh,
      struct ether_arp *arp_frame,
      unsigned char destination_mac[6]);
    ARPHeader* build_arp_request(
        struct ether_header *eh,
        struct ether_arp *rp_frame,
        const unsigned char hop_ip[4]);
    uint16_t checksum(unsigned char* addr, int len);
    std::string get_ip_str(unsigned char[4]);
    bool host_in_lookup_table(std::string host, std::unordered_map<std::string, std::string>);
    int Start(std::string lookup);
};
} // namespace router

#endif
