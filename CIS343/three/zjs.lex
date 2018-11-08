%{
  #include <stdio.h>
  #include <stdlib.h>
  #include "zjs.h"
  #include "zjs.tab.h"
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
[0-9]+          { return yylval.intValue = atoi(yytext); return INT; }
[0-9]+\.[0-9]+  { return yylval.floatValue = atof(yytext); return FLOAT; }
[' '\t\n]       ;
.               { printf("Unknown character: %s\n", yytext); }

%%
