%{
  #include <stdio.h>
%}

%option noyywrap
%option yylineno

%%

end             { printf("END"); }
;               { printf("END_STATEMENT"); }
line            { printf("LINE"); }
point           { printf("POINT"); }
circle          { printf("CIRCLE"); }
rectangle       { printf("RECTANGLE"); }
set_color       { printf("SET_COLOR"); }
[0-9]+          { printf("INTEGER"); }
[0-9]+\.[0-9]+  { printf("FLOATING POINT"); }
[' '\t\n]       ;
.               { printf("Error on line: %d\n", yylineno); }

%%

int main(int argc, char** argv) {
  yylex();
}
