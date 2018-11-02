#include "./include/ChatClient.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <sys/socket.h>
#include <tuple>

namespace client {
void* ChatClient::worker(void* args) {
  ChatClient c;
  ChatClient::thread t;
  ChatClient::std_message s;
  std::memcpy(&t, args, sizeof(ChatClient::thread));

  std::string data;
  while (data != "/quit" && data != "kicked\n") {
    recv(t.socket, &s, sizeof(ChatClient::std_message), 0);
    // Decrypt our message
    c.decrypt(s.cipher, t.key, s.iv, data);
    if (data == "kicked") {
      std::cout << "OHH HO HO HOOO YOU HAVE BEEN KICKED MY BOY" << std::endl;
      exit(0);
      return nullptr;
    }
    std::cout << " <<< " << data << std::endl;
  }

  return nullptr;
}


void ChatClient::err() {
  ERR_print_errors_fp(stderr);
  abort();
}

std::string ChatClient::encrypt(
    const std::string& plaintext,
    unsigned char* key,
    unsigned char* iv,
    const std::string& cipher) {
  EVP_CIPHER_CTX *ctx;
  int cipher_len = sizeof(cipher.c_str());

  if (!(ctx = EVP_CIPHER_CTX_new())) {
    err();
  }
  if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, key, iv)) {
    err();
  }

  if (1 != EVP_EncryptUpdate(
        ctx,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(cipher.c_str())),
        &cipher_len,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(plaintext.c_str())),
        static_cast<int>(sizeof(plaintext.c_str())))) {
    err();
  }

  if (1 != EVP_EncryptFinal_ex(
        ctx,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(cipher.c_str())) + cipher_len,
        &cipher_len)) {
    err();
  }
  EVP_CIPHER_CTX_free(ctx);

  return plaintext;
}

std::string ChatClient::decrypt(
    const std::string& plaintext,
    unsigned char* key,
    unsigned char* iv,
    std::string cipher) {
  EVP_CIPHER_CTX *ctx;
  int plaintext_len = sizeof(plaintext.c_str());
  if (!(ctx = EVP_CIPHER_CTX_new())) {
    err();
  }

  if (1 != EVP_DecryptUpdate(
        ctx,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(plaintext.c_str())),
        &plaintext_len,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(cipher.c_str())),
        static_cast<int>(sizeof(cipher.c_str())))) {
    err();
  }

  if (1 != EVP_DecryptFinal_ex(
        ctx,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(plaintext.c_str())) + plaintext_len,
        &plaintext_len)) {
    err();
  }
  EVP_CIPHER_CTX_free(ctx);

  return plaintext;
}
} // namespace client
