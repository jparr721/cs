#ifndef INCLUDE_ROUTER_ICMPHEADER_HPP
#define INCLUDE_ROUTER_ICMPHEADER_HPP

#include <netinet/ether.h>

namespace router {
class ICMPHeader {
  public:
    uint8_t type;
    uint8_t code;
    uint8_t checksum;
};
} // namespace router

#endif
