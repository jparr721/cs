# Lab 8
Jarred Parr Alexander Fountain

1. This is a buffer overrun issue. Because the proper memory was not properly allocated, the length of the string overflows into other parts of memory that may not have been originally designated to the program. As a result, this causes problems. On a more high-scale project, this could be catastrophic and open the code to injection issues that would potentially cause harm to the target computer or exploit the software as a whole. The error in this file begins with the malloc of size 16. The problem here is that when you put in `notarealusername` the length is too large, and on the `scanf` line you see the data get placed into that memory incorrectly as a result.

**Corrected Code**
```C

```
