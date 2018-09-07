#include "operations.h"

void addProduct(product ** l, product node) {
	product * newProduct;
	newProduct = malloc(sizeof(product));

	strncpy(newProduct->pName, node.pName, 20);
	strncpy(newProduct->qUnit, node.qUnit, 20);
	strncpy(newProduct->pUnit, node.pUnit, 20);
	newProduct->quantity = node.quantity;
	newProduct->price = node.price;
	newProduct->next = *l;
	*l = newProduct;

	puts("New Item added \n");
	printf("Name: %s\n", newProduct->pName);
	printf("Quantity: %f%s\n", newProduct->quantity, newProduct->qUnit);
	printf("Price: %f %s\n\n\n", newProduct->price, newProduct->pUnit);
	displayMenu();
}

void purchase(product* l, char name[], float quantity) {
	if (l != NULL) {
		if (quantity <= l->quantity) {
			float newQuantity = l->quantity - quantity;
			l->quantity = newQuantity;
			l->profit += quantity * l->price;
			printf("Purchased %f %s of %s\n", quantity, l->qUnit, l->pName);
			printf("Profits: %f", l->profit);
		} else {
			puts("We do not have enough in stock, sorry :(");
		}
	} else {
		puts("Could not find that product, please check our stock!");
	}
	displayMenu();
}

void checkPrice(product* l, char name[]) {
	product* current = l;
	while(current != NULL) {
		if (strcmp(current->pName, name) == 0) {
			printf("The price of %s is: %f%s\n\n", current->pName, current->price, current->pUnit);
		} else {
			puts("Sorry, that product doesn't exist in the store");
		}
		current = current->next;
	}
	displayMenu();
}

void showProducts(product* l) {
	product* current = l;

	puts("Items currently in store:\n");
	 while(current != NULL) {
		printf("--%s\n\n", current->pName);
		current = current->next;
	}
	displayMenu();
}

void cleanUpProduct(product* l, char name[]) {
	product* current = l;

	if(strcmp(current->pName, l->pName) == 0) {
		list = current->next;	
	} else {
		while(current->next != NULL) {
			if(strcmp(current->next->pName, l->pName) == 0) {
				product* removed = current->next;
				if(removed->next != NULL) {
					current->next = removed->next;
					puts("Item removed successfully!");
				} else {
					current->next = NULL;
				}
				break;
			}
			current = current->next;
		}
	}
	displayMenu();
}

void findProduct(product* l, char name[]) {
	product* current = l;

	while(current != NULL) {
		if(strcmp(current->pName, name) == 0) {
			printf("Product: %s\n\nQuantity: %f%s\n\nPrice: %f%s\n", current->pName, current->quantity, current->qUnit, current->price, current->pUnit);
			break;
		} else {
			puts("Error, could not find product with that name");
		}
		current = current->next;
	}
	displayMenu();
}

void showInv(product* l) {
	product* current = l;
	puts("Full item inventory: \n");
	
	while(current != NULL) {
		printf("------------%s-------------\n", current->pName);
		printf("Quantity: %f%s\nPrice: %f%s\n", current->quantity, current->qUnit, current->price, current->pUnit);
		current = current->next;
	}
	displayMenu();
}

void save(char outf[]) {
	puts("Now saving...");
	FILE *f = fopen("storelog.txt", "w");
	if (f == NULL) {
		printf("Error saving file!\n");
		exit(1);
	}

	product* current = list;

	while(current != NULL) {
		fprintf(f, "%s\n", current->pName);
		fprintf(f, "%f\n", current->quantity);
		fprintf(f, "%s\n", current->qUnit);
		fprintf(f, "%f\n", current->price);
		fprintf(f, "%s\n", current->pUnit);
		current = current->next;
	}
	fclose(f);
}

void load(char inf[]) {
	puts("Now loading...");
	FILE *f = fopen("storelog.txt", "r");
	if (f == NULL) {
		printf("Error loading file!\n");
		exit(1);
	}

	char line[20];

	product p;
	
	int i = 0;

	while(fgets(line, sizeof(line), f)) {
		line[strlen(line) -1] = '\0';
		if (i == 0)
			strcpy(p.pName, line);
		else if (i == 1)
			p.quantity = strtof(line, NULL);
		else if (i == 2)
			strcpy(p.qUnit, line);
		else if (i == 3)
			p.price = strtof(line, NULL);
		else if (i == 4) {
			strcpy(p.pUnit, line);
			i = -1;
			addProduct(&list, p);
		}
		i++;
	}
	fclose(f);
}

void done() {
	save("storelog.txt");
	printf("Profits: %f\n", list->profit);
	puts("Thanks for stopping by!");
	exit(0);
}

