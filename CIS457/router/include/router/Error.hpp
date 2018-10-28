#ifndef ERROR_HPP
#define ERROR_HPP

#include <cstdint>
#include <cstring>
#include <iostream>

namespace router {
class Error {
  public:
    const uint8_t TYPE_TTL = 11;
    const uint8_t TYPE_UNREACHABLE = 3;
    const uint8_t CODE_ZERO = 0;
    const uint8_t CODE_ONE = 1;

    unsigned char* time_to_live(unsigned char* icmp_buffer);
    uint16_t checksum(unsigned char* addr, int len);
    unsigned char* create_error(
        unsigned char* type,
        unsigned char* code,
        unsigned char checksum[16],
        unsigned char* ipheader);
};
} // namespace router

#endif
