#ifndef INCLUDE_ROUTER_ROUTER_HPP
#define INCLUDE_ROUTER_ROUTER_HPP

#include <vector>
/* #include <router/ARPHeader.hpp> */
#include "./ARPHeader.hpp"

namespace router {
class Router {
  public:
    Router() = default;
    ~Router() = default;

    ARPHeader build_arp_reply(
      struct ether_header *eh,
      struct ether_arp *arp_frame,
      uint8_t destination_mac);
    ARPHeader build_arp_request(
      std::vector<ARPHeader> r,
      std::vector<struct ether_header> eh,
      std::vector<struct ether_arp> arp_frame,
      uint8_t hop_ip[4]);
    void get_dest_mac(
      std::vector<struct ifaddrs> ifaddr,
      std::vector<struct ifaddrs> tmp,
      uint8_t arp_tpa[4],
      int socket,
      uint8_t dmac[6]);
    void get_src_mac(
      std::vector<struct ifaddrs> ifaddr,
      std::vector<struct ifaddrs> tmp,
      uint8_t if_ip[4],
      int socket,
      uint8_t dmac[6]);

    int Start();
};
} // namespace router

#endif
