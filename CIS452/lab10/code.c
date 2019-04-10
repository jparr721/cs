#include <stdio.h>
#include <string.h>
#include <Windows.h>

char* getState(int state){
    if(state == 4096) { // 0x1000
        return "COMMIT";
    } else if (state == 65536) { // 0x10000
        return "FREE";
    } else if (state == 8192) { // 0x2000
        return "RESERVE";
    } else {
        return "ERROR";
    }
}

int main(){
    struct _SYSTEM_INFO info;
    GetSystemInfo(&info);
    printf("Page Size: %d\n", info.dwPageSize);
    char* c = malloc(1000000);
    struct _MEMORY_BASIC_INFORMATION memInfo;
    SIZE_T length;
    VirtualQuery(c, &memInfo, length);
    char* state = getState(memInfo.State);
    printf("%s\n", state);
    free(c);
    VirtualQuery(c, &memInfo, length);
    state = getState(memInfo.State);
    printf("%s\n", state);
}
