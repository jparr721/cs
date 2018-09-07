#include <iostream>
#include "bst.h"

BinarySearchTree::BinarySearchTree() {
    root = new BinaryNode();
    root->data = 23;
    root->left = nullptr;
    root->right = nullptr;
}

void BinarySearchTree::insert(const int & key, BinaryNode *&root) {
    if (key < root->data) {
        if (root->left == nullptr) {
            root->left = new BinaryNode();
            root->left->data = key;
        }
        insert(key, root->left);
    } else if (root->data < key){
        if (root->right == nullptr) {
            root->right = new BinaryNode();
            root->right->data = key;
        }
        insert(key, root->right);
    }
}

void BinarySearchTree::makeEmpty(BinaryNode *& root) {
    if (root != nullptr){
        makeEmpty(root->left);
        makeEmpty(root->right);
        delete root;
    }
    root = nullptr;
}

void BinarySearchTree::preorder(BinaryNode * root) {

    if (root == nullptr) {
        std::cout << "The tree is empty!" << std::endl;
    }

    if (root != nullptr) {
        std::cout << root->data << " " << std::flush;

        if (root->left != nullptr) {
            preorder(root->left);
        }

        if (root->right != nullptr)
            preorder(root->right);

    }


}

void BinarySearchTree::inorder(BinaryNode * root) {
    if (root == nullptr) {
        std::cout << "The tree is empty!" << std::endl;
    }
    else {
        if (root->left != nullptr)
            inorder(root->left);

        std::cout << root->data << " " << std::flush;

        if (root->right != nullptr)
            inorder(root->right);

    }
}


void BinarySearchTree::postorder(BinaryNode * root) {
    if (root == nullptr)
        std::cout<< "The tree is empty!" << std::endl;
    else {
        if (root->left != nullptr)
            postorder(root->left);

        if (root->right != nullptr)
            postorder(root->right);

        std::cout << root->data << " " << std::flush;
    }



}
