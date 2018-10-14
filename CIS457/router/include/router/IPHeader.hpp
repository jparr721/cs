#ifndef INCLUDE_ROUTER_IPHEADER_HPP
#define INCLUDE_ROUTER_IPHEADER_HPP

#include <netinet/ether.h>

namespace router {
class IPHeader {
  public:
    uint8_t ihl:4, version:4;
    uint8_t dif_services;
    uint8_t len; // Datagram length.
    uint8_t id; // Datagram identity.
    uint8_t flag_offset;
    uint8_t ttl; // Time to live.
    uint8_t protocol; // Protocol type.
    uint8_t checksum;
    unsigned char src_ip[4]; // Sender IP address.
    unsigned char dest_ip[4]; // Target IP address.
};
} // namespace router

#endif
