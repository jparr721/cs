#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <netinet/in.h>
#include <iostream>
#include <string>
#include <sstream>
#include <sys/socket.h>
#include <pthread.h>
#include <unistd.h>

class Server {
  public:
    const int MAXDATASIZE = 4096;
    int get_port();
    auto input_handler();
    static void* handle_client(void* arg);
    int RunServer();
};

int Server::get_port() {
  std::cout << "Enter the port to connect to: " << std::endl;
  std::string port = "";
  std::getline(std::cin, port);

  return std::stoi(port);
}

auto Server::input_handler() {
  std::cout << " >>> " << std::endl;
  std::string message = "";
  std::getline(std::cin, message);

  // Must be handled as a char*
  return message.c_str();
}

void* Server::handle_client(void* arg) {
  int clientsocket = *(int*)arg;
  while (true) {
    char line[4096];
    recv(clientsocket, line, 4096, 0);
    std::cout << "<<< " << line << std::endl;
    send(clientsocket, line, strlen(line) + 1, 0);
    close(clientsocket);
  }

  return nullptr;
}

int Server::RunServer() {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    std::cerr << "Error creating socket" << std::endl;
    return EXIT_FAILURE;
  }

  int port = this->get_port();
  struct sockaddr_in server, client;
  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = INADDR_ANY;

  if (bind(sockfd, (struct sockaddr*) &server, sizeof(server)) < 0) {
    std::cerr << "Failed to bind to socket" << std::endl;
    return EXIT_FAILURE;
  }

  // Build later
  while (true) {
    socklen_t sin_size = sizeof client;

    int clientsocket = accept(sockfd, reinterpret_cast<sockaddr*>(&client), &sin_size);
    // Create the child thread
    pthread_t child;

    // Make the thread continue on the socket
    pthread_create(&child, nullptr, Server::handle_client, &clientsocket);

    // When thread is done, dump the return value
    pthread_detach(child);
  }

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  Server s;
  s.RunServer();

  return EXIT_SUCCESS;
}
