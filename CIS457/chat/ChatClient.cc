#include "./include/ChatClient.hpp"
#include "./include/User.hpp"
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

void* client::ChatClient::client_handler(void* args) {
  the::Succ SLIP_SLOP;
  ChatClient::thread t;
  ChatClient::std_message s;
  std::memcpy(&t, args, sizeof(ChatClient::thread));

  std::string data;
  while (data != "/quit" && data != "kicked") {

    char buf[4096];

    recv(t.socket, buf, 4096, 0);
    // Decrypt our message
    //data = SLIP_SLOP.decrypt(t.key, s.iv, s.cipher);
    data = std::string(buf);
    if (data == "kicked") {
      std::cout << "OHH HO HO HOOO YOU HAVE BEEN KICKED MY BOY" << std::endl;
      exit(0);
      return nullptr;
    }
    std::cout << " <<< " << data << std::endl;
  }

  return nullptr;
}
namespace client {

in_addr_t ChatClient::handle_host() {
  std::cout << "Please enter the host you would like to connect to: " << std::flush;
  std::string host = "";
  std::getline(std::cin, host);

  return inet_addr(host.c_str());
}

int ChatClient::handle_port() {
  std::cout << "Please enter the port to connect from: " << std::flush;
  std::string port = "";
  std::getline(std::cin, port);

  return std::stoi(port);
}


std::string ChatClient::handle_input() {
  std::cout << " >>> " << std::flush;
  std::string message = "";
  std::getline(std::cin, message);

  return message;
}

int ChatClient::RunClient() {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  the::Succ DIVINE_SUCC;

  if (sockfd < 0) {
    std::cerr << "Error creating the socket" << std::endl;
    return EXIT_FAILURE;
  }
  int port = handle_port();
  in_addr_t host = handle_host();
  unsigned char key[32];

  struct sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = host;

  socklen_t sin_size = sizeof server;

  int c = connect(sockfd, reinterpret_cast<sockaddr*>(&server), sin_size);
  if (c < 0) {
    std::cerr << "Error connecting" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "You connected." << std::endl;

  std::string username = "";
  ChatClient::symmetric_key_message skm;
  int taken = 1;
  bool kicked = false;

  //ERR_load_crypto_strings();
  //OpenSSL_add_all_algorithms();
  //OPENSSL_config(nullptr);
  //RAND_bytes(key, 32);


  FILE* rsa_public_key = std::fopen("rsa_pub.pem", "rb");
  EVP_PKEY *public_key;
  public_key = PEM_read_PUBKEY(rsa_public_key, nullptr, nullptr, nullptr);

  // Set our key with the rsa encryption
  skm.encrypted_key = DIVINE_SUCC.rsa_encrypt(
      std::string(reinterpret_cast<char*>(key)),
      public_key);

  // Send the key to the server so it can decrypt the messages
  sendto(sockfd, &skm, sizeof(ChatClient::symmetric_key_message), 0, reinterpret_cast<sockaddr*>(&server), sin_size);

  std::cout << "Please enter a username" << std::endl;
  std::getline(std::cin, username);
  sendto(sockfd, username.c_str(), username.length(), 0, reinterpret_cast<sockaddr*>(&server), sin_size);

  // Allocate space on the heap
  ChatClient::thread *t = new ChatClient::thread;
  std::memcpy(&t->socket, &sockfd, sizeof(int));

  pthread_t child;
  pthread_create(&child, NULL, client_handler, t);
  pthread_detach(child);

  while (true) {
    ChatClient::std_message s_message;
    std::string message = handle_input();

    if (message == "/quit\n" && !kicked) {
      break;
    }

    //RAND_pseudo_bytes(s_message.iv, 16);
    // Encrpyt our message
    //s_message.cipher = DIVINE_SUCC.encrypt(key, s_message.iv, s_message.cipher);

    // Send it along
    send(sockfd, message.c_str(), message.length(), 0);
  }

  close(sockfd);

  return EXIT_SUCCESS;
}

} // namespace client


int main() {
  client::ChatClient cs;
  cs.RunClient();

  return EXIT_SUCCESS;
}
