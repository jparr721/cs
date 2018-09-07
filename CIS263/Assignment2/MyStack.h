//
// Created by Xiang Cao on 9/9/2017.
//

#ifndef HW2_MYSTACK_H
#define HW2_MYSTACK_H

#include<vector>
#include<iostream>
#include<curl/curl.h>
using namespace std;

template <typename Object>
class MyStack
{
private:
    vector<Object> elements;
public:
    void push(const Object&);   //push an item in the stack
    void pop(); //pop an item from the stack
    Object top() const; //return the item on the top of the stack, no actual pop() happens
    bool empty() const; //whether the stack is empty?
    int GetSize() const;    //get the number of elements in the stack
};


#endif //HW2_MYSTACK_H
