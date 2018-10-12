/* #include <router/Router.hpp> */
#include "../include/router/Router.hpp"
#include "../include/router/TableLookup.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <ifaddrs.h>
#include <iostream>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <sys/socket.h>
#include <string>
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
    // PROTOCOL LENGT
    r.ea.ea_hdr.ar_pln = sizeof(in_addr_t);
    // OP
    r.ea.ea_hdr.ar_op = htons(ARPOP_REPLY);
    // ETHERNET HEADER
    r.eh = *eh;

    return r;
  }

  int Router::Start() {
    TableLookup routeTable("r1-table.txt");

//    routeTable.getRoute("10.0.0.0/16");

    int packet_socket;

    struct ifaddrs *ifaddr, *tmp;
    fd_set interfaces;
    FD_ZERO(&interfaces);

    if (getifaddrs(&ifaddr) == -1) {
      std::cerr << "getifaddrs machine broke" << std::endl;
      return EXIT_FAILURE;
    }

    for (tmp = ifaddr; tmp != nullptr; tmp=tmp->ifa_next) {
      if (tmp->ifa_addr->sa_family == AF_PACKET) {
        std::cout << "Interface: " << tmp->ifa_name << std::endl;

        if (!strncmp(&(tmp->ifa_name[3]), "eth1", 4)) {
          std::cout << "Creating socket on interface: " << tmp->ifa_name << std::endl;

          packet_socket = socket(AF_INET, SOCK_RAW, htons(ETH_P_ALL));
          if (packet_socket < 0) {
            std::cerr << "socker machine broke" << std::endl;
            return EXIT_FAILURE;
          }

          if (bind(packet_socket, tmp->ifa_addr, sizeof(struct sockaddr_ll)) == -1) {
            std::cerr << "bind machine broke" << std::endl;
          }
        }
      }
    }

    std::cout << "My body is ready" << std::endl;

    while (1) {
      char buffer[1500];
      struct sockaddr_ll recvaddr;
      socklen_t recvaddrlen = sizeof(struct sockaddr_ll);

      int n = recvfrom(packet_socket, buffer, 1500, 0, (struct sockaddr*) &recvaddr, &recvaddrlen);
      if (n < 0) {
        std::cerr << "recvfrom machine broke" << std::endl;
      }

      if (recvaddr.sll_pkttype == PACKET_OUTGOING)
        continue;

      std::cout << "Got a " << n << " byte packet" << std::endl;
    }

    freeifaddrs(ifaddr);

    return EXIT_SUCCESS;
  }
} // namespace router
