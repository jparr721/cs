#include <stdio.h>
#include <stdlib.h>

void unpack(short empId) {
	// short mask = 0x001;
	// short jobType = encodedValue & mask;
	// encodedValue = encodedValue >> 1;
	//
	// mask = 0x000f;
	// short position = (encodedValue >> 1) & mask;
	// mask = 0x00f;
	// short gender = (encodedValue >> 4) & mask;
	// mask = 0x03ff;
	// short gNum = (encodedValue >> 2) & mask;
	short jobMask = 0x0001;
    short jobType = empId & jobMask;

    empId = empId >> 1;
    short positionMask = 0x000f;
    short position = empId & positionMask;

    empId = empId >> 3;
    short genderMask = 0x0003;
    short gender = empId & genderMask;

    empId = empId >> 2;
    short gNumMask = 0x03ff;
    short gNum = empId & gNumMask;

	printf("G Number: %d \n", gNum);
	switch(gender) {
		case 0:
			printf("Gender: Female\n");
			break;
		case 1:
			printf("Gender: Male\n");
			break;
		case 2:
			printf("Gender: Not Tell\n");
			break;
		default:
			break;
	}
	switch(position) {
		case 0:
			printf("Position: Professor\n");
			break;
		case 1:
			printf("Position: Lecturer\n");
			break;
		case 2:
			printf("Position: Admin\n");
			break;
		case 3:
			printf("Position: IT Staff\n");
			break;
		case 4:
			printf("Position: Visitor\n");
			break;
		case 5:
			printf("Position: Student\n");
			break;
		case 6:
			printf("Position: Dean\n");
			break;
		case 7:
			printf("Position: President\n");
	}
	switch(jobType) {
		case 0:
			printf("Job Type: Full Time\n");
			break;
		case 1:
			printf("Job Type: Part Time\n");
			break;
		default:
			break;
	}
}

unsigned short pack(short gNum, short gender, short position, short jobType) {
	unsigned short emp = 0;
	emp = (emp | gNum) << 2;
	emp = (emp | gender) << 3;
	emp = (emp | position) << 1;
	emp = (emp | jobType);
	return emp;
}


int main(int argc, char* argv[]) {
	unsigned short encodedValue = 0;
	if (argc < 2) {
		puts("Error please try again. invalid number of arguments!");
	} else if(argc == 2) {
		short encodedValue =atoi(argv[1]);
		unpack(encodedValue);
	} else if(argc == 5) {
		short gNum = atoi(argv[1]);
		short gender = atoi(argv[2]);
		short position = atoi(argv[3]);
		short jobType = atoi(argv[4]);
		encodedValue = pack(gNum, gender, position, jobType);
		printf("%d", encodedValue);
	}

}
