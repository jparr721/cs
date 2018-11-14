#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <tuple>

namespace client {
class Client {
  public:
    int Run();

    struct thread {
      int socket;
      unsigned char key[32];
    };

    struct std_msg {
      std::string cipher;
      unsigned char iv[16];
    };

    struct sym_key_msg {
      unsigned char* encr_key;
    };
  private:
    const int MAXDATASIZE = 4096;
    static void* handler(void* args);

    int handle_port();
    in_addr_t handle_host();
    std::string handle_input();

    std::tuple<sockaddr_in, int> initialize_client();
};
} // namespace client
