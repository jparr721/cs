#include <stdio.h>
#include <stdlib.h>
#include "functions.h"

char* key;
char* ascii  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
char* cipher = "ZYXWVUTSRQPONMLKJIHGFEDCBA";

char* removeDuplicates(char input[]) {
		char* output = malloc(strlen(input)+1);
		int output_index = 0;
		int i = 0;
		for (i = 0; i < strlen(input); i++) {
				char* result = strchr(output, input[i]);
				int index = (int) (result - output);
				if (index < 0) {
						output[output_index] = input[i];
						output_index++;
				}
		}
		output[strlen(output)] = '\0';
		strcpy(input, output);
		free(output);
		return input;
}

char* generateCipher() {
		char* temp = malloc(strlen(key) + strlen(cipher));
		strcpy(temp, key);
		strcat(temp, cipher);
		return removeDuplicates(temp);
}

char* encrypt(char text[]) {
		puts("Now encrypting... \n\n");
		char* newCipher = generateCipher();
		char out[strlen(text)-1];
		int i = 0;
		for (i = 0; i < strlen(text); i++){
				int keyVal = text[i];
				if(keyVal != 32 && (keyVal - 65) >= 0) {
						out[i] = newCipher[keyVal - 65];
				}
				else {
						out[i] = ' ';
				}
		}
		int difference = strlen(out) - strlen(text);
		out[strlen(out) - difference] = '\0';
		strcpy(text, out);
		return text;
}

char* decrypt(char text[]){
		puts("Now decrypting...\n\n");
		char* newCipher = generateCipher();
		char out[strlen(text)-1];
		int i = 0;
		for (i = 0; i < strlen(text); i++){
				int keyVal = text[i];
				if (keyVal != 32 && (keyVal - 65) >= 0) {
						char* res = strchr(newCipher, text[i]);
						int index = (int)(res - newCipher);
						out[i] = ascii[index];
				} else {
						out[i] = ' ';
				}
		}
		int diff = strlen(out)-strlen(text);
		out[strlen(out)-diff] = '\0';
		strcpy(text, out);
		return text;

}

char* processInput(char file[]) {
		FILE* inFile;
		inFile = fopen(file, "r");

		if (inFile == NULL){
				printf("Error, could not open file");
				exit(1);
		}

		char* buffer = malloc(255);
		fgets(buffer, 255, inFile);
		fclose(inFile);
		return buffer;
}

void processOutput(char file[], char text[]){
		FILE* outFile;
		outFile = fopen(file, "w");

		if(outFile == NULL){
				printf("Error, could not open file");
				exit(1);
		}
		fprintf(outFile, text);
		fclose(outFile);
}
