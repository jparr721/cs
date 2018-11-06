#ifndef CHAT_CHAT_SERVER_HPP
#define CHAT_CHAT_SERVER_HPP

#include "../include/User.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace server {
class ChatServer {
  public:
    const std::string version = "0.1.0";
    const int MAXDATASIZE = 4096;
    int taken = 0;

    int RunServer();

    struct thread {
      // Whatever socket it's bound to
      int socket;

      std::string username;

      struct sockaddr_in client;

      // To decrypt our goodies
      unsigned char* key;
      
      ChatServer* instance;

      pthread_t* child;
    };

    struct std_message {
      std::string cipher;

      // Initialization vector
      unsigned char iv[16];
    };

    bool get_admin();
    void set_admin(bool admin);

    // The /list command
    void list_users();

    // The /broadcast command
    void broadcast(const std::string& message);

    // Currently connected users
    std::vector<thread> users;

  private:
    bool is_admin;
    bool check_admin(const std::string& pass);
    static void* server_handler(void* args);

    int handle_port();

    std::string handle_input(std::string prompt);

    // Making auto for bool return or string
    std::string extract_command(const std::string& input) const;

    std::string admin_password = "1234";

};
} // namespace server

#endif
