#include <iostream>
#include "parr.cpp"     //in order to make the template work correctly, intentionally include "MyStack.cpp" instead of "MyStack.h"
using namespace std;

int main() {
    int i;
    MyStack<int> Stack_int;
    MyStack<float> Stack_float;
    MyStack<string> Stack_string;

    //testing the stack_int
    cout<<"-----testing the stack_int-----"<<endl;
    for(i=0;i<5;i++)
    {
        Stack_int.push(i);
        cout<<Stack_int.top()<<" ";
    }
    cout<<"have been pushed!"<<endl;
    for(i=0;i<5;i++)
    {
        cout<<Stack_int.top()<<" ";
        Stack_int.pop();
    }
    cout<<"have been popped!"<<endl;
    Stack_int.pop();    //an additional pop() intentionally

    //testing the stack_float
    cout<<"-----testing the stack_float-----"<<endl;
    Stack_float.pop();
    Stack_float.push(1.5);
    Stack_float.push(4.9);
    Stack_float.push(8.3);
    cout<<"Current stack size is "<<Stack_float.GetSize()<<endl;
    cout<<Stack_float.top()<<" has been popped!"<<endl;
    Stack_float.pop();
    cout<<"Current stack size is "<<Stack_float.GetSize()<<endl;

    //testing the stack_string
    cout<<"-----testing the stack_string-----"<<endl;
    cout<<"Current stack size is "<<Stack_string.GetSize()<<endl;
    Stack_string.push("abc");
    Stack_string.push("efg");
    cout<<"Current stack size is "<<Stack_string.GetSize()<<endl;
    cout<<"The top of current stack is "<<Stack_string.top()<<endl;
    if (Stack_string.empty())
    {
        cout<<"Current stack is empty!"<<endl;
    }
    else
    {
        cout<<"Current Stack is not empty!"<<endl;
    }

    return 0;
}