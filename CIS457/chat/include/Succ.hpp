#ifndef THE_SUCC
#define THE_SUCC

#include <string>
#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>

namespace the {
  class Succ {
    public:
      void err();
      unsigned char* rsa_encrypt(
          const std::string& in,
          EVP_PKEY *key);
      unsigned char* rsa_decrypt(
          const std::string& in,
          EVP_PKEY *key);
      std::string encrypt(
          unsigned char* key,
          unsigned char* iv,
          const std::string& cipher);
      std::string decrypt(
          unsigned char* key,
          unsigned char* iv,
          const std::string& cipher);
  };
} // namespace the
#endif
