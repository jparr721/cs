#pragma once


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define N 20

struct product {
	char pName[N];
	float quantity;
	char qUnit[N];
	float price;
	char pUnit[N]; // because it sounds like G UNIT BOIIII
	float profit;
	struct product* next;
};

typedef struct product product;

extern int choice;
product* list;

void addProduct(product**, product);

void purchase(product*, char[], float);

void checkPrice(product*, char[]);

void showProducts(product*);

void cleanUpProduct(product*, char[]);

void findProduct(product*, char[]);

void showInv(product*);

void done();

void load(char inf[]);

void save(char outf[]);

void displayMenu();
