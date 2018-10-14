#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <router/Router.hpp>
#include <router/TableLookup.hpp>
#include <sys/ioctl.h>
#include <sys/socket.h>
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

  // Adapted from here: http://www.microhowto.info/howto/get_the_ip_address_of_a_network_interface_in_c_using_siocgifaddr.html
  uint8_t router::Router::get_dest_mac(
    struct ifaddrs *ifaddr,
    struct ifaddrs *tmp,
    uint8_t *arp_tpa,
    int socket
    ){
    struct ifreq ifr;
    uint8_t destination_mac_addr[6];

    for (tmp = ifaddr; tmp != nullptr; tmp = tmp->ifa_next) {
      if (tmp->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in* sa = (struct sockaddr_in*) tmp->ifa_addr;
        // Extract IP of captured packet
        char* ip_addr = inet_ntoa(sa->sin_addr);
        char arp_addr[50];
        // Build arp target IP to compare
        sprintf(arp_addr, "%d.%d.%d.%d", arp_tpa[0], arp_tpa[1], arp_tpa[2], arp_tpa[3]);

        if (std::strcmp(ip_addr, arp_addr) == 0) {
          // prepare our ifreq struct to use ioctl
          ifr.ifr_addr.sa_family = AF_INET;
          strcpy(ifr.ifr_name, tmp->ifa_name);
          if (ioctl(socket, SIOCGIFADDR, &ifr) == -1) {
            std::cerr << "Failed to run ioctl" << std::endl;
            return -1;
          }
          std::memcpy(destination_mac_addr, ifr.ifr_hwaddr.sa_data, 6);
        }
      }
    }

    return *destination_mac_addr;
  }

  uint8_t router::Router::get_src_mac(
    struct ifaddrs *ifaddr,
    struct ifaddrs *tmp,
    uint8_t if_ip,
    int socket,
    uint8_t destination_mac
    ){
    return destination_mac;
  }

  int Router::Start(std::string routing_table) {
    TableLookup routeTable(routing_table);

    int packet_socket;

    struct ifaddrs *ifaddr, *tmp;
    fd_set interfaces;
    FD_ZERO(&interfaces);

    if (getifaddrs(&ifaddr) == -1) {
      std::cerr << "getifaddrs machine broke" << std::endl;
      return EXIT_FAILURE;
    }

    for (tmp = ifaddr; tmp != nullptr; tmp = tmp->ifa_next) {
      if (tmp->ifa_addr->sa_family == AF_PACKET) {
        std::cout << "Interface: " << tmp->ifa_name << std::endl;

        if (!strncmp(&(tmp->ifa_name[3]), "eth1", 4)) {
          std::cout << "Creating socket on interface: " << tmp->ifa_name << std::endl;

          packet_socket = socket(AF_INET, SOCK_RAW, htons(ETH_P_ALL));
          if (packet_socket < 0) {
            std::cerr << "socket machine broke" << std::endl;
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
