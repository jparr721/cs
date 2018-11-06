%{
  #include <stdio.h>
  #include <stdlib.h>
  #include "zjs.h"
%}

%option noyywrap
%option yylineno

%%

end             return END;
;               return END_STATEMENT;
line            return LINE;
point           return POINT;
circle          return CIRCLE;
rectangle       return RECTANGLE;
set_color       return SET_COLOR;
[0-9]+          return INTEGER;
[0-9]+\.[0-9]+  return FLOAT;
[' '\t\n]       return IGNORE;
.               return ERR;

%%

int main(int argc, char** argv) {
  yylex();
}
