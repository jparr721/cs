/* #include <router/Router.hpp> */
/* #include <router/TableLookup.hpp> */
// Commmented out to avoid compiler warnings
#include "../include/router/Router.hpp"
#include "../include/router/TableLookup.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <ifaddrs.h>
#include <iostream>
#include <net/if.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace router {
  ARPHeader Router::build_arp_reply(
      struct ether_header *eh,
      struct ether_arp *arp_frame,
      uint8_t destination_mac
      ) {
    ARPHeader r;
    // SOURCE MAC FORMAT
    r.ea.ea_hdr.ar_hrd = htons(ARPHRD_ETHER);
    // SOURCE MAC LENGTH
    r.ea.ea_hdr.ar_hln = ETHER_ADDR_LEN;
    // TARGET MAC
    std::memcpy(r.ea.arp_tha, &arp_frame->arp_sha, 6);
    // TARGET PROTOCOL
    std::memcpy(r.ea.arp_tpa, &arp_frame->arp_spa, 4);
    // SOURCE MAC ADDRESS
    std::memcpy(static_cast<uint8_t*>(r.ea.arp_sha), &destination_mac, 6);
    // PROTOCOL
    r.ea.ea_hdr.ar_pro = htons(ETH_P_IP);
    // PROTOCOL LENGTH
    r.ea.ea_hdr.ar_pln = sizeof(in_addr_t);
    // OP
    r.ea.ea_hdr.ar_op = htons(ARPOP_REPLY);
    // ETHERNET HEADER
    r.eh = *eh;

    return r;
  }

  ARPHeader Router::build_arp_request(
      struct ether_header *eh,
      struct ether_arp *arp_fame,
      uint8_t hop_ip
      ){
    ARPHeader r;
    // SOURCE MAC FORMAT
    r.ea.ea_hdr.ar_hrd = htons(ARPHRD_ETHER);
    // PROTOCOL
    r.ea.ea_hdr.ar_pro = htons(ETH_P_IP);
    // SOURCE MAC LENGTH
    r.ea.ea_hdr.ar_hln = ETHER_ADDR_LEN;
    // SOURCE PROTOCOL LENGTH
    r.ea.ea_hdr.ar_pln = sizeof(in_addr_t);
    // OP
    r.ea.ea_hdr.ar_op = htons(ARPOP_REQUEST);
    // ETHERNET HEADER
    r.eh = *eh;

    return r;
  }

  uint8_t router::Router::get_dest_mac(
    struct ifaddrs *ifaddr,
    struct ifaddrs *tmp,
    uint8_t arp_tpa,
    int socket
    ){
    struct ifreq ifr;

    for (tmp = ifaddr; tmp != nullptr; tmp = tmp->ifa_next) {
      if (tmp->ifa_addr->sa_family == AF_INET) {

      }
    }
  }

  uint8_t router::Router::get_src_mac(
    struct ifaddrs *ifaddr,
    struct ifaddrs *tmp,
    uint8_t if_ip,
    int socket,
    uint8_t destination_mac
    ){

  }

  int router::Router::Start() {
    TableLookup routeTable("r1-table.txt");

    return EXIT_SUCCESS;
  }
} // namespace router
