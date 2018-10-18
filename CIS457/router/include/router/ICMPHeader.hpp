#ifndef INCLUDE_ROUTER_ICMPHEADER_HPP
#define INCLUDE_ROUTER_ICMPHEADER_HPP

#include <netinet/ether.h>

namespace router {
class ICMPHeader {
  public:
    uint8_t type;
    uint8_t code;
    uint16_t checksum;
		uint16_t id;
		uint16_t seq;
		uint32_t data;
};
} // namespace router

#endif
