#include "./include/ChatServer.hpp"

#include "./include/Succ.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <sys/socket.h>
#include <tuple>
#include <unistd.h>


namespace server {
void* ChatServer::server_handler(void* args) {
  the::Succ SKOOMA_HIGH;
  ChatServer::std_message s;
  ChatServer::thread t;
  ChatServer cs;

  std::memcpy(&t, args, sizeof(ChatServer::thread));
  // Data sent over the wire
  char data[4096];

  while (true) {
    if (std::string(data) == "/quit") {
      std::cout << "Shutting down the server connection to user: " << t.username << std::endl;
      close(t.socket);
      // Break this worker
      break;
    }

    int r = recv(t.socket, &s, sizeof(ChatServer::std_message), 0);
    if (r < 0) {
      break;
    }

    std::string message = SKOOMA_HIGH.decrypt(t.key, s.iv, s.cipher);
    std::cout << " <<< " << message << std::endl;

    std::string command = cs.extract_command(message);

    if (command == "/list") {
      cs.list_users();
    } else if (command == "/broadcast") {

    }
  }
}

void ChatServer::broadcast() {

}

void ChatServer::list_users() {
  std::cout << "Listing users..." << std::endl;
  if (!users.empty()) {
    for (const auto &user : users) {
      std::cout << user.username << std::endl;
    }

    return;
  }

  std::cout << "No clients found." << std::endl;
}

std::string ChatServer::extract_command(const std::string& input) const {
  std::string command = "no_command";
  for (const auto v : input) {
    if  (v == '/') {
      // Strips everything before the command delimeter
      command = input.substr(input.find('/') + 1);
      return command;
    }
    return command;
  }
}

int ChatServer::RunServer() {
  return EXIT_SUCCESS;
}
} // namespace server
