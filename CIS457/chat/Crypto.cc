/*This code is built with heavy copying from examples from the openssl
  wiki, along with examples from openssl man pages*/

#include "./include/Crypto.hpp"

#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>
#include <string.h>


void Crypto::handleErrors(void)
{
  ERR_print_errors_fp(stderr);
  abort();
}


int Crypto::rsa_encrypt(unsigned char* in, size_t inlen, EVP_PKEY *key, unsigned char* out){ 
  EVP_PKEY_CTX *ctx;
  size_t outlen;
  ctx = EVP_PKEY_CTX_new(key, NULL);
  if (!ctx)
    handleErrors();
  if (EVP_PKEY_encrypt_init(ctx) <= 0)
    handleErrors();
  if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING) <= 0)
    handleErrors();
  if (EVP_PKEY_encrypt(ctx, NULL, &outlen, in, inlen) <= 0)
    handleErrors();
  if (EVP_PKEY_encrypt(ctx, out, &outlen, in, inlen) <= 0)
    handleErrors();
  return outlen;
}

int Crypto::rsa_decrypt(unsigned char* in, size_t inlen, EVP_PKEY *key, unsigned char* out){ 
  EVP_PKEY_CTX *ctx;
  size_t outlen;
  ctx = EVP_PKEY_CTX_new(key,NULL);
  if (!ctx)
    handleErrors();
  if (EVP_PKEY_decrypt_init(ctx) <= 0)
    handleErrors();
  if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING) <= 0)
    handleErrors();
  if (EVP_PKEY_decrypt(ctx, NULL, &outlen, in, inlen) <= 0)
    handleErrors();
  if (EVP_PKEY_decrypt(ctx, out, &outlen, in, inlen) <= 0)
    handleErrors();
  return outlen;
}

int Crypto::encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *key,
	unsigned char *iv, unsigned char *ciphertext){
  EVP_CIPHER_CTX *ctx;
  int len;
  int ciphertext_len;
  if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();
  if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv))
    handleErrors();
  if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len))
    handleErrors();
  ciphertext_len = len;
  if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) handleErrors();
  ciphertext_len += len;
  EVP_CIPHER_CTX_free(ctx);
  return ciphertext_len;
}

int Crypto::decrypt(unsigned char *ciphertext, int ciphertext_len, unsigned char *key,
	    unsigned char *iv, unsigned char *plaintext){
  EVP_CIPHER_CTX *ctx;
  int len;
  int plaintext_len;
  if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();
  if(1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv))
    handleErrors();
  if(1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len))
    handleErrors();
  plaintext_len = len;
  if(1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) handleErrors();
  plaintext_len += len;
  EVP_CIPHER_CTX_free(ctx);
  return plaintext_len;
}
/*
int main(void){
  unsigned char *pubfilename = "RSApub.pem";
  unsigned char *privfilename = "RSApriv.pem";
  unsigned char key[32];
  unsigned char iv[16];
  unsigned char *plaintext =
    (unsigned char *)"This is a test string to encrypt.";
  unsigned char ciphertext[1024];
  unsigned char decryptedtext[1024];
  int decryptedtext_len, ciphertext_len;
  OpenSSL_add_all_algorithms();
  RAND_bytes(key,32);
  RAND_bytes(iv,16);
  EVP_PKEY *pubkey, *privkey;
  FILE* pubf = fopen(pubfilename,"rb");
  pubkey = PEM_read_PUBKEY(pubf,NULL,NULL,NULL);
  unsigned char encrypted_key[256];
  int encryptedkey_len = rsa_encrypt(key, 32, pubkey, encrypted_key);
  ciphertext_len = encrypt (plaintext, strlen ((char *)plaintext), key, iv,
                            ciphertext);
  printf("Ciphertext is:\n");
  BIO_dump_fp (stdout, (const char *)ciphertext, ciphertext_len);

  FILE* privf = fopen(privfilename,"rb");
  privkey = PEM_read_PrivateKey(privf,NULL,NULL,NULL);
  unsigned char decrypted_key[32];
  int decryptedkey_len = rsa_decrypt(encrypted_key, encryptedkey_len, privkey, decrypted_key); 
  
  decryptedtext_len = decrypt(ciphertext, ciphertext_len, decrypted_key, iv,
			      decryptedtext);
  decryptedtext[decryptedtext_len] = '\0';
  printf("Decrypted text is:\n");
  printf("%s\n", decryptedtext);
  EVP_cleanup();
  ERR_free_strings();
  return 0;
}
*/
