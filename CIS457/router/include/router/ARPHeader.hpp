#ifndef INCLUDE_ROUTER_ARPHEADER_HPP
#define INCLUDE_ROUTER_ARPHEADER_HPP

#include <netinet/ether.h>

namespace router {
class ARPHeader {
  public:
    struct ether_header eh;
    struct ether_arp ea;
};
} // namespace router

#endif
