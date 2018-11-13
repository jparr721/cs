#include "./include/Client.hpp"
#include "./include/Crypto.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace client {
std::tuple<sockaddr_in, int> Client::initialize_client() {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    std::cerr << "Error creating the socket" << std::endl;
    exit(EXIT_FAILURE);
  }

  int port = handle_port();
  in_addr_t host = handle_host();

  sockaddr_in server;
  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = host;
  socklen_t sin_size = sizeof server;

  int c = connect(sockfd, reinterpret_cast<sockaddr*>(&server), sin_size);
  if (c < 0) {
    std::cout << "Failed to establish server connection, exiting" << std::endl;
    exit(EXIT_FAILURE);
  }
  OpenSSL_add_all_algorithms();

  return std::make_tuple(server, sockfd);
}
int Client::Run() {
  auto params = initialize_client();
  sockaddr_in server = std::get<0>(params);
  int sockfd = std::get<1>(params);
  socklen_t sin_size = sizeof server;

  std::cout << "Connection established" << std::endl;
  std::string username;
  Client::sym_key_msg skm;
  int kicked = false;
  unsigned char key[32];
  unsigned char iv[16];
  RAND_bytes(key, 32);
  RAND_bytes(iv, 16);

  yep::Crypto crypto;

  std::cout << "Constructing public key for data transmission" << std::endl;

  FILE* pubkey_file = fopen("rsa_pub.pem", "rb");
  EVP_PKEY *pubkey = PEM_read_PUBKEY(pubkey_file, nullptr, nullptr, nullptr);
  std::cout << "Key constructed" << std::endl;

  unsigned char encrypted_key[256];
  std::memset(encrypted_key, 0, 256);
  std::cout << "Encrypting our key to share with the server" << std::endl;
  int encrypted_key_len = crypto.rsa_encrypt(key, 32, pubkey, encrypted_key);
  std::cout << "Sharing encrypted key with the server" << std::endl;

  int send_size = sendto(sockfd, &encrypted_key, encrypted_key_len,
      0, reinterpret_cast<sockaddr*>(&server), sin_size);

  std::cout << "Please enter your username: " << std::flush;
  std::getline(std::cin, username);
  std::cout << "Informaing the server of your choice" << std::endl;
  send_size = sendto(sockfd, username.c_str(), username.length(),
      0, reinterpret_cast<sockaddr*>(&server), sin_size);

  std::cout << "Creating a new chat thread" << std::endl;
  auto *t = new Client::thread;
  std::memcpy(&t->socket, &sockfd, sizeof(int));
  std::memcpy(&t->key, &key, 32);

  pthread_t client;
  pthread_create(&client, nullptr, handler, t);
  pthread_detach(client);

  for (;;) {
    Client::std_msg smsg;
    std::string message = handle_input();

    if (message == "/quit" && !kicked) {
      std::cout << "Quit message received, exiting" << std::endl;
      break;
    }

    unsigned char* plaintext = const_cast<unsigned char*>(
        reinterpret_cast<const unsigned char*>(message.c_str()));
    unsigned char miv[16];

    std::memset(miv, 0, 16);
    unsigned char cipher[1024];
    int cipher_len = crypto.encrypt(plaintext, message.size(), key, miv, cipher);

    send(sockfd, cipher, cipher_len, 0);
  }
  std::cout << "Thanks for stopping by!" << std::endl;

  close(sockfd);

  return EXIT_SUCCESS;
}

void* Client::handler(void* args) {
  yep::Crypto crypto;
  Client::thread t;
  Client::std_msg s;
  std::memcpy(&t, args, sizeof(Client::thread));

  std::string data;
  while (data != "/quit" && data != "kicked") {
    char buf[4096];
    int input = recv(t.socket, buf, 4096, 0);

    unsigned char* cipher = reinterpret_cast<unsigned char*>(buf);
    unsigned char miv[16];
    unsigned char plaintext[1024];

    int plaintext_len = crypto.decrypt(cipher, input, t.key, miv, plaintext);

    data = std::string(reinterpret_cast<char*>(plaintext));
    if (data  == "kicked") {
      std::cout << "You have been kicked, sorrry :(" << std::endl;
      exit(0);
      return nullptr;
    }

    std::cout << " <<< " << data << "\n <<< " << std::endl;
  }

  return nullptr;
}

in_addr_t Client::handle_host() {
  std::cout << "Please enter the host you would like to connect to: " << std::flush;
  std::string host;
  std::getline(std::cin, host);

  return inet_addr(host.c_str());
}

int Client::handle_port() {
  std::cout << "Please enter the port to connect from: " << std::flush;
  std::string port;
  std::getline(std::cin, port);

  return std::stoi(port);
}

std::string handle_input() {
  std::cout << " >>> " << std::flush;
  std::string message;
  std::getline(std::cin, message);

  return message;
}

} // namespace client
