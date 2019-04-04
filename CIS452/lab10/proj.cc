#include <Windows.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <string>


std::string convert_memory_state(int state) {
	if (state == 4096) {
		return "Committed";
	}
	else if (state == 0x10000) {
		return "Free";
	}
	else if (state == 0x2000) {
		return "Reserve";
	}
	else {
		return "Broken";
	}
}

int main() {
	SYSTEM_INFO sys_info;

	GetSystemInfo(&sys_info);

	// Query system and determine page size
	std::cout << "PageSize[Bytes]: " << sys_info.dwPageSize << std::endl;


	// Malloc 2^20 bytes dynamically without needing to free
	auto big_val = std::make_shared<char*>(new char[2 << 20]);
	MEMORY_BASIC_INFORMATION mbi;
	auto size = VirtualQuery(&big_val, &mbi, sizeof(mbi));

	// Report new memory usage
	std::cout << "Memory State: " << convert_memory_state(mbi.State) << std::endl;

	// Delete shared ptr's memory space
	big_val.reset();

	// Repeat the system state
	MEMORY_BASIC_INFORMATION mbi2;
	auto size2 = VirtualQuery(&big_val, &mbi2, sizeof(mbi2));
	std::cout << "Memory State: " << convert_memory_state(mbi2.State) << std::endl;

	getchar();

	return 0;
}
