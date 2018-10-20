%{
  #include <stdio.h>
%}

%%

[St|Av|Dr|Ln]             { printf("STREET TYPE %s\n", yytext); }
[A-Z][a-z]+ [A-Z][a-Z]+   { printf("NAME %s\n", yytext); }
[A-Z]{2}                  { printf("STATE %s\n", yytext); }
[0-9]{5}(-[0-9]{4})?      { printf("Zipcode: %s\n", yytext); }

/* If space or tab or newline, if nore it */
[' '\t\n]            ;
%%

int main(int argc, char** argv) {
  yylex();
}
