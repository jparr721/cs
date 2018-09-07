#include <iostream>
#include "bst.h"

using namespace std;

int main() {
    int key[15] = {23, 12, 9, 45, 8, 37, 41, 5, 3, 18, 32, 14, 49, 52, 1};
    BinarySearchTree t;
    int i;
    cout<<"inserting ";
    for (i=0;i<15;i++)
    {
        t.insert(key[i]);
        cout<<key[i]<<" ";
    }
    cout<<endl<<"Preorder: ";
    t.preorder();
    cout<<endl<<"Inorder: ";
    t.inorder();
    cout<<endl<<"Postorder ";
    t.postorder();
    cout<<endl;
    t.makeEmpty();
    t.preorder();
    t.inorder();
    t.postorder();
    return 0;
}