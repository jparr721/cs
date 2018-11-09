%{
	#include <stdio.h>
	#include <stdlib.h>
	#include "zoomjoystrong.h"
	#include "zoomjoystrong.tab.h"
%}

%option noyywrap

%%

end               { return(END); }
;                 { return(END_STATEMENT); }
point             { return(POINT); }
line              { return(LINE); }
circle            { return(CIRCLE); }
rectangle         { return(RECTANGLE); }
set_color         { return(SET_COLOR); }
[0-9]+         	 	{ yylval.intValue = atoi(yytext); return INT; }
[0-9]+\.[0-9]+		{ yylval.floatValue = atof(yytext); return FLOAT; }
[' '\t\n]         ;
.                 { printf("Unknown character: %s\n", yytext); }

%%
