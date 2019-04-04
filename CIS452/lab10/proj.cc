#include <Windows.h>
#include <stdlib.h>
#include <memory>
#include <iostream>

int main()
{
    SYSTEM_INFO sys_info;

    GetSystemInfo(&sys_info);

    // Query system and determine page size
    std::cout << "PageSize[Bytes]: " << sys_info.dwPageSize << std::endl;

    // Malloc 2^20 bytes dynamically without needing to free
    std::unique_ptr<char*> big_val(new char(2>>20));
    GetSystemInfo(&sys_info);

    // Report new memory usage
    std::cout << "PageSize[Bytes]: " << sys_info.dwPageSize << std::endl;

    // Delete shared ptr's memory space
    big_val.reset();

    // Repeat the system state
    GetSystemInfo(&sys_info);
    std::cout << "PageSize[Bytes]: " << sys_info.dwPageSize << std::endl;

    return 0;
}
