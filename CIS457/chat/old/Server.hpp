#pragma once

#include <arpa/inet.h>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <vector>

namespace server {
class Server {
  public:
    int Run();

    struct thread {
      int socket;
      std::string username;
      sockaddr_in client;
      unsigned char key[32];
      Server* instance;
    };

    struct std_msg {
      std::string cipher;
      unsigned char iv[16];
    };

    std::pair<std::string, int> encrypt_string(std::string input, unsigned char key[32]);

    void list_users();

    void broadcast(const std::string& message);
    std::string extract_command(const std::string& input) const;

    bool check_admin(const std::string& pass);
    // Use a smart pointer to destroy null refs
    std::vector<thread> users;

  private:
    bool is_admin;
    const std::string ADMIN_PASSWORD = "1234";

    static void* handler(void* args);

    int handle_port();
    std::string handle_input();
};
} // namespace server
