#ifndef INCLUDE_ROUTER_ETHHEADER_HPP
#define INCLUDE_ROUTER_ETHHEADER_HPP

#include <netinet/ether.h>
namespace router {
class ETHHeader {
  public:
    uint8_t eth_dest[6];
    uint8_t eth_src[6];
    uint8_t eth_type;
};
} // namespace router
#endif
