#include <iostream>
#include <string>
#include "house.h"
using namespace std;



house::house() {
    this -> color = "unknown";
    this -> price = -1;
    this -> num_rooms = -1;
}

house::house(const string &ColorValue, int PriceValue, int Num_roomsValue) {
    this -> color = ColorValue;
    this -> price = PriceValue;
    this -> num_rooms = Num_roomsValue;
}

void house::SetColor(const string &ColorValue) {
    this -> color = ColorValue;
}

string house::GetColor() const {
    return color;
}

void house::SetPrice(int PriceValue) {
    this -> price = PriceValue;
}

int house::GetPrice() const {
    return price;
}

void house::SetNum_rooms(int Num_roomsValue) {
    this -> num_rooms = Num_roomsValue;
}

int house::GetNum_rooms() const {
    return num_rooms;
}

void house::PrintInfo() const {
    cout << color <<  " " << price <<  " " << num_rooms << endl;
}
