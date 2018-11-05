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
      std::string outgoing = cs.handle_input("What do you want to say?: ");
    }
  }
}

std::string ChatServer::handle_input(std::string prompt = " >>> ") {
  std::cout << prompt << std::flush;
  std::string message = "";
  std::getline(std::cin, message);

  return message;
}

void ChatServer::broadcast(const std::string& message) {
  the::Succ succ;
  for (const auto &user : users) {
    ChatServer::std_message outgoing;
    RAND_pseudo_bytes(outgoing.iv, 16);
    outgoing.cipher = succ.encrypt(
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(user.key)),
        outgoing.iv,
        message);

    send(user.socket, message.c_str(), sizeof(ChatServer::std_message), 0);
  }
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
