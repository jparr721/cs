#include<iostream>
#include<vector>
#include"MyStack.h"
using namespace std;

template <typename Object>
void MyStack<Object>::push(const Object & obj) {
    elements.push_back(obj);
}

template <typename Object>
void MyStack<Object>::pop() {
    if (!elements.size() == 0)
        elements.pop_back();
    else
        cout << "The Stack is now empty! No item is popped!" << endl;

}

template <typename Object>
Object MyStack<Object>::top() const {
   return elements.back();
}

template <typename Object>
bool MyStack<Object>::empty() const {
    return elements.size() == 0;
}

template <typename Object>
int MyStack<Object>::GetSize() const {
    return elements.size();
}

