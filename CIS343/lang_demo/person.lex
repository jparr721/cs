%{
#include <stdio.h>
%}

%%

[A-Z][a-z]+
// phone number
\([0-9]{3}\)-[0-9]{3}-[0-9]{4}
// day emd in day
[A-z[a-z]+day
%%

int main(int argc, char** argv) {
  yylex();
}
