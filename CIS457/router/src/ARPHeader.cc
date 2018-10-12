#include <router/ARPHeader.hpp>


namespace router {
  ARPHeader::ARPHeader() {
    this->eh = nullptr;
    this->ea = nullptr;
  }
} // namespace router
