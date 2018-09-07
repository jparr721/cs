#ifndef PROJ1_FUNCTIONS_H
#define PROJ1_FUNCTIONS_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

extern char* key;
extern char* ascii;
extern char* cipher;

char* encrypt(char[]);

char* decrypt(char[]);

char* processInput(char file[]);

char* generateCipher();

char* removeDuplicates(char*);

void processOutput(char file[], char text[]);

#endif //PROJ1_FUNCTIONS_H
