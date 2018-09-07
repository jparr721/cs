#include <iostream>
#include "house.h"
using namespace std;

int main() {
    house A;
    cout<<"House A: "<<A.GetColor()<<" "<<A.GetPrice()<<" "<<A.GetNum_rooms()<<endl;
    A.SetColor("White");
    A.SetPrice(150000);
    A.SetNum_rooms(3);
    cout<<"Now, House A: ";
    A.PrintInfo();

    house B("Grey", 200000, 4);
    cout<<"House B: "<<B.GetColor()<<" "<<B.GetPrice()<<" "<<B.GetNum_rooms()<<endl;

    house *C = new house;
    cout<<"House C: "<<C->GetColor()<<" "<<C->GetPrice()<<" "<<C->GetNum_rooms()<<endl;
    C->SetColor("Yellow");
    C->SetPrice(180000);
    C->SetNum_rooms(3);
    cout<<"Now, House C: ";
    C->PrintInfo();

    house *D = new house("Red", 120000, 2);
    cout<<"House D: "<<D->GetColor()<<" "<<D->GetPrice()<<" "<<D->GetNum_rooms()<<endl;

    delete C;
    delete D;

    return 0;
}