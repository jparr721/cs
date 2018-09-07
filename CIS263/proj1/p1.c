#include "functions.h"


int main(int argc, char* argv[]) {
	char cmd = *argv[1]; // either e or d
	char* file;
	char cipher[26];
	char* in;
	char* out;
	char* text;
	key = argv[2];
	switch(cmd) {
		case 'e':
			text = processInput(argv[3]);
			in = encrypt(text);
			printf("After encryption, the output is: %s\n", in);
			processOutput(argv[4], in);
			break;
		case 'd':
			text = processInput(argv[3]);
			out = decrypt(text);
			printf("After decryption, the output is: %s\n", out);
			processOutput(argv[4], out);
			break;
		default:
			printf("Instructions \n\n");
            		printf("You're probably here because you made a mistake \n\n");
			printf("Invocation format: \n");
			printf("./p e/d key infile outfile\n\n");
			printf("You can also use \"make enc\" to encrypt, and \"make dec\" to decrypt\n\n");
	}
return 0;
}



