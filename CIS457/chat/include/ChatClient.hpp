#ifndef CHAT_CHATCLIENT_HPP
#define CHAT_CHATCLIENT_HPP

#include <openssl/rsa.h>
#include <string>

namespace client {
class ChatClient {
  public:
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
      unsigned char encrypted_key[256];
    };

    void setKicked(bool kicked);
    bool getKicked();
  private:
    const int MAXDATASIZE = 4096;
    static void* worker(void* args);
    bool kicked = false;

    void err();
    int rsa_encrypt(
        std::string in,
        size_t len,
        EVP_PKEY *key,
        std::string out) const;
    int rsa_decrypt(
        std::string in,
        size_t len,
        EVP_PKEY *key,
        std::string out) const;
    std::string encrypt(
        const std::string& plaintext,
        unsigned char* key,
        unsigned char* iv,
        const std::string& cipher);
    std::string decrypt(
        const std::string& plaintext,
        unsigned char* key,
        unsigned char* iv,
        std::string cipher);
};
} // namespace client

#endif
