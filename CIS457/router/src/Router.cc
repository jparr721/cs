/* #include <router/Router.hpp> */
#include "../include/router/Router.hpp"

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
  int Router::Start() {
    int packet_socket;

    struct ifaddrs *ifaddr, *tmp;

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
