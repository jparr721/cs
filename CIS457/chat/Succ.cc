#include "./include/Succ.hpp"

void the::Succ::err() {
  ERR_print_errors_fp(stderr);
  abort();
}

std::string the::Succ::encrypt(
    unsigned char* key,
    unsigned char* iv,
    const std::string& cipher) {
  std::string plaintext;
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

std::string the::Succ::decrypt(
    unsigned char* key,
    unsigned char* iv,
    const std::string& cipher) {
  std::string plaintext;
  EVP_CIPHER_CTX *ctx;
  int plaintext_len = sizeof(plaintext.c_str());
  if (!(ctx = EVP_CIPHER_CTX_new())) {
    err();
  }

  if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, key, iv)) {
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

unsigned char* the::Succ::rsa_decrypt(
    const std::string& in,
    EVP_PKEY *key) {
  EVP_PKEY_CTX *ctx;
  size_t outlen = 0;
  unsigned char* output;

  ctx = EVP_PKEY_CTX_new(key, nullptr);

  if (!ctx) {
    err();
  }

  if (EVP_PKEY_decrypt_init(ctx) <= 0) {
    err();
  }

  if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING) <= 0) {
    err();
  }

  if (EVP_PKEY_decrypt(
        ctx,
        nullptr,
        &outlen,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(in.c_str())),
        in.length()) <= 0) {
    err();
  }

  if (EVP_PKEY_encrypt(
        ctx,
        output,
        &outlen,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(in.c_str())),
        in.length()) <= 0) {
    err();
  }

  return output;
}

unsigned char* the::Succ::rsa_encrypt(
    const std::string& in,
    EVP_PKEY *key) {
  EVP_PKEY_CTX *ctx;
  size_t outlen = 0;
  unsigned char* output;

  ctx = EVP_PKEY_CTX_new(key, nullptr);

  if (!ctx) {
    err();
  }

  if (EVP_PKEY_encrypt_init(ctx) <= 0) {
    err();
  }

  if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING) <= 0) {
    err();
  }

  if (EVP_PKEY_encrypt(
        ctx,
        nullptr,
        &outlen,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(in.c_str())),
        in.length()) <= 0) {
    err();
  }

  if (EVP_PKEY_encrypt(
        ctx,
        output,
        &outlen,
        const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(in.c_str())),
        in.length()) <= 0) {
    err();
  }

  return output;
}
