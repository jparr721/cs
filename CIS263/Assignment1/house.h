#pragma once
#include <string>
using namespace std;

class house
{
private:
    string color;  //color of the house
    int price;   //price of the house
    int num_rooms;   //number of rooms in the house
public:
    house();
    house(const string &ColorValue, int PriceValue, int Num_roomsValue);
    void SetColor(const string &ColorValue);  //set the member variable "color"
    string GetColor() const; //return the value of the member variable "color"
    void SetPrice(int PriceValue);  //set the member variable "price"
    int GetPrice() const;  //return the value of the member variable "price"
    void SetNum_rooms(int Num_roomsValue);  //set the member variable "num_rooms"
    int GetNum_rooms() const;  //return the value of the member variable "num_rooms"
    void PrintInfo() const; //show color, price and num_rooms in the output screen
};