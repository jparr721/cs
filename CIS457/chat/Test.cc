#include "./include/Test.hpp"
#include "./include/Crypto.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <sys/socket.h>
#include <tuple>
#include <unistd.h>

int main() {
  Crypto JEFF;
  
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
  FILE* pubf = fopen("rsa_pub.pem","rb");
  pubkey = PEM_read_PUBKEY(pubf,NULL,NULL,NULL);
  unsigned char encrypted_key[256];
  int encryptedkey_len = JEFF.rsa_encrypt(key, 32, pubkey, encrypted_key);
  ciphertext_len = JEFF.encrypt(plaintext, strlen ((char *)plaintext), key, iv,
                            ciphertext);
  printf("Ciphertext is:\n");
  BIO_dump_fp (stdout, (const char *)ciphertext, ciphertext_len);

  FILE* privf = fopen("rsa_priv.pem","rb");
  privkey = PEM_read_PrivateKey(privf,NULL,NULL,NULL);
  unsigned char decrypted_key[32];
  int decryptedkey_len = JEFF.rsa_decrypt(encrypted_key, encryptedkey_len, privkey, decrypted_key); 
  
  decryptedtext_len = JEFF.decrypt(ciphertext, ciphertext_len, decrypted_key, iv,
			      decryptedtext);
  decryptedtext[decryptedtext_len] = '\0';
  printf("Decrypted text is:\n");
  printf("%s\n", decryptedtext);
  EVP_cleanup();
  ERR_free_strings();
  return 0;
}
