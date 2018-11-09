%{
  #include <stdio.h>
  #include <stdlib.h>
  #include "zoomjoystrong.h"
  int yyerror(const char* msg);
  extern char* yytext;
int yylex();
%}

%union {
  int intValue;
  float floatValue;
  char* stringValue;
}

%token END
%token END_STATEMENT
%token LINE
%token POINT
%token CIRCLE
%token RECTANGLE
%token SET_COLOR
%token <intValue> INT
%token <floatValue> FLOAT

%%
program: list_of_expr end;
list_of_expr: expr
            | expr list_of_expr;
expr: line
    | point
    | circle
    | rectangle
    | set_color
    ;

line: LINE INT INT INT INT END_STATEMENT
{
  if($2 >= 0 && $2 <= WIDTH &&  $3 >= 0 && $3 <= HEIGHT) {
    line($2,$3,$4,$5);
  } else {
    printf("Points must be between 0 and %d for width and 0 and %d for height", WIDTH, HEIGHT);
  }
};

point: POINT INT INT END_STATEMENT
{
  if ($2 >= 0 && $2 <= WIDTH && $3 >= 0 && $3 <= HEIGHT) {
    point($2, $3);
  } else {
    printf("Points must be between 0 and %d for width and 0 and %d for height\n", WIDTH, HEIGHT);
  }
};

circle: CIRCLE INT INT INT END_STATEMENT
{
  if($2 >= 0 && $2 <= WIDTH && $3 >= 0 && $3 <= HEIGHT && $4 >= 0){
    circle($2,$3,$4);
  } else {
    printf("Points must be between 0 and %d for width and 0 and %d for height and positive for radius\n", WIDTH, HEIGHT);
  }
};

rectangle: RECTANGLE INT INT INT INT END_STATEMENT
{
  if ($2 >= 0 && $2 <= WIDTH && $3 >= 0 && $3 <= HEIGHT) {
    rectangle($2, $3, $4, $5);
  } else {
    printf("Points must be between 0 and %d for width and 0 and %d for height\n", WIDTH, HEIGHT);
  }
};

set_color: SET_COLOR INT INT INT END_STATEMENT
{
  if ($2 >= 0 && $2 <= 255 && $3 >= 0 && $3 <= 255 && $4 >= 0 && $4 <= 255) {
    set_color($2, $3, $4);
  } else {
    printf("RGB Values must be betweem 0 - 255\n");
  }
};

end: END END_STATEMENT
{
  finish();
  return 0;
};

%%

extern FILE* yyin;

int main(int argc, char** argv) {
  setup();
  yyin = fopen(argv[1], "r");
  yyparse();
  return 0;
}

int yyerror(const char* msg) {
  printf("%s\n", msg, yytext);
  return 1;
}

