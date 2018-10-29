#include "../include/router/Error.hpp"
#include "../include/router/Router.hpp"

#include <arpa/inet.h>

namespace router {
unsigned char* Error::time_to_live(unsigned char* icmp_buffer) {
  unsigned char data[84];
  std::memcpy(&data, icmp_buffer, sizeof(data));

  uint8_t ttl;
  std::memcpy(&ttl, icmp_buffer + 14 + 9, sizeof(uint8_t));

  ttl--;

  if (ttl <= 0) {
    unsigned char csum[16];

    std::memcpy(&csum, icmp_buffer + 50, sizeof(uint8_t));
    // Bug on this line
    icmp_buffer = this->create_error((unsigned char*) this->TYPE_TTL, (unsigned char*) this->CODE_ZERO, csum, data);
    return icmp_buffer;
  } else {
    // remake the checksum
    uint16_t check[2];
    std::memcpy(check, icmp_buffer + 24, sizeof(uint16_t));
    uint16_t new_checksum = htons(this->checksum((unsigned char*)check, 2));

    std::memcpy(icmp_buffer + 24, &new_checksum, sizeof(uint16_t));
    return icmp_buffer;
  }
}

uint16_t Error::checksum(unsigned char *addr, int len) {
    int nleft = len;
    const uint16_t *w = (const uint16_t *)addr;
    uint32_t sum = 0;
    uint16_t answer = 0;

    /*
     * Our algorithm is simple, using a 32 bit accumulator (sum), we add
     * sequential 16 bit words to it, and at the end, fold back all the
     * carry bits from the top 16 bits into the lower 16 bits.
     */
    while (nleft > 1)  {
      sum += *w++;
      nleft -= 2;
    }

    /* mop up an odd byte, if necessary */
    if (nleft == 1) {
      *(unsigned char *)(&answer) = *(const unsigned char *)w ;
      sum += answer;
    }

    /* add back carry outs from top 16 bits to low 16 bits */
    sum = (sum & 0xffff) + (sum >> 16);
    sum += (sum >> 16);
    /* guaranteed now that the lower 16 bits of sum are correct */

    answer = ~sum;              /* truncate to 16 bits */
    return answer;
}

unsigned char* Error::create_error(
    unsigned char* type,
    unsigned char* code,
    unsigned char checksum[16],
    unsigned char* ipheader
    ) {
  uint32_t zeros = 0x000;
  unsigned char* error;

  std::memcpy(error, type, sizeof(uint8_t));
  std::memcpy(error + sizeof(uint8_t), checksum, sizeof(uint16_t));
  std::memcpy(error + 2 * sizeof(uint8_t), checksum, sizeof(uint16_t));
  std::memcpy(error + 8 * sizeof(uint8_t), ipheader, sizeof(ipheader));

  return error;
}
} // namespace router
