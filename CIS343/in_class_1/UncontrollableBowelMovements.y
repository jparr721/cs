%{
#include <stdio.h>
void yyerror(const char* msg);
%}

%union { int i; char str; }

%error-verbose

%token G
%token X
%token Y
%token END
%token ZERO
%token EOL
%token INT

%type<i> INT
%type<str> G
%type<str> X
%type<str> Y
%type<str> ZERO

%%
program: list_of_expr
       ;

list_of_expr: expr
            | list_of_expr expr
;

expr: G INT X INT Y INT Z INT EOL
    | ZERO EOL
;
%%

int main(int argc, char** argv) {

}

void yyerror(const char* msg) {
  fprintf(stderr, "ITS BROKEN AHHHHHHHH: %s\n", msg);
}
