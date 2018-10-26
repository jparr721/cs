%{
  #include <stdio.h>
%}

%option noyywrap

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
.               { printf("This aint it chief"); }

%%

int main(int argc, char** argv) {
  yylex();
}
