#ifndef CHAT_CHATCLIENT_HPP
#define CHAT_CHATCLIENT_HPP

#include "./include/User.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <openssl/rsa.h>
#include <string>
#include <sys/socket.h>


namespace client {
class ChatClient {
  public:
    const std::string version = "0.1.0";

    int RunClient();
    // Holds thread specific shit
    struct thread {
      // Whatever socket it's bound to
      int socket;
      // To decrypt our goodies
      unsigned char key[32];
    };

    struct std_message {
      std::string cipher;
      unsigned char iv[16];
    };

    struct symmetric_key_message {
      unsigned char* encrypted_key;
    };

    void setKicked(bool kicked);
    bool getKicked();
  private:
    const int MAXDATASIZE = 4096;
    static void* worker(void* args);
    bool kicked = false;

    void err();

    int handle_port();
    in_addr_t handle_host();

    std::string handle_input();

    unsigned char* rsa_encrypt(
        const std::string& in,
        EVP_PKEY *key);
    unsigned char* rsa_decrypt(
        const std::string& in,
        EVP_PKEY *key);
    std::string encrypt(
        const std::string& plaintext,
        unsigned char* key,
        unsigned char* iv,
        const std::string& cipher);
    std::string decrypt(
        const std::string& plaintext,
        unsigned char* key,
        unsigned char* iv,
        const std::string& cipher);

};
} // namespace client

#endif
