#pragma once

#include <cstring>
#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>

namespace yep {
class Crypto {
  public:
    void handleErrors();
    int rsa_encrypt(unsigned char* in, size_t inlen, EVP_PKEY *key, unsigned char* out);
    int rsa_decrypt(unsigned char* in, size_t inlen, EVP_PKEY *key, unsigned char* out);
    int encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *key,
          unsigned char *iv, unsigned char *ciphertext);
    int decrypt(unsigned char *ciphertext, int ciphertext_len, unsigned char *key,
          unsigned char *iv, unsigned char *plaintext);
};
} // namespace crypto
