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

int ChatServer::handle_port() {
  std::cout << "Please enter the port for the server: " << std::flush;
  std::string port = "";
  std::getline(std::cin, port);

  return std::stoi(port);
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

  int sock = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(handle_port());

  if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    std::cout << "Failed to bind socket." << std::endl;
    return EXIT_FAILURE;
  }  

  while (true) {
    // listen for a new client connection
    listen(sock, 5);

    struct sockaddr_in client;
    socklen_t sin_size = sizeof(client);

    int clientsocket = accept(sock, reinterpret_cast<sockaddr*>(&client), &sin_size);

    if (clientsocket > -1) {
      std::cout << "Client conected" << std::endl;
    } else {
      std::cout << "oh no no no no" << std::endl;
    }

    struct thread t;    
    t.socket = clientsocket;
    char buf[1024];

    // receive the key
    recv(clientsocket, &t.key, sizeof(t.key), 0);

    // receive the username in the buf for no reason
    recv(clientsocket, &buf, 1024, 0);
    t.username = (std::string) buf;

    std::cout << t.key << std::endl;
    std::cout << "username: " + t.username << std::endl;

    // acknowledge username
    int ack = 1;
    send(clientsocket, &ack, sizeof(int), 0); 
    
    ChatServer::users.push_back(t);

    pthread_t client_r;
    pthread_create(&client_r, nullptr, ChatServer::server_handler, &t);
    pthread_detach(client_r);
    
  }
  
  
  return EXIT_SUCCESS;
}

} // namespace server

int main() {
  server::ChatServer cs;
  cs.RunServer();

  return EXIT_SUCCESS;
}
