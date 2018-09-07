//
// Created by Xiang Cao on 9/25/17.
//

#ifndef HW3_BST_H
#define HW3_BST_H

class BinarySearchTree {
private:
    struct BinaryNode {
        int data;
        BinaryNode *left;
        BinaryNode *right;
    };
    BinaryNode *root;
    void insert(const int &, BinaryNode *&); //insert an element into BST
    void preorder(BinaryNode *);
    void postorder(BinaryNode *);
    void inorder(BinaryNode *);
    void makeEmpty(BinaryNode *&);  //delete the entire BST

public:
    BinarySearchTree();

    void insert(const int &key)   //insert an element into BST
    {
        insert(key, root);
    }

    void preorder()
    {
        preorder(root);
    }

    void inorder()
    {
        inorder(root);
    }

    void postorder()
    {
        postorder(root);
    }

    void makeEmpty()   //delete the entire BST
    {
        makeEmpty(root);
    }
};

#endif //HW3_BST_H