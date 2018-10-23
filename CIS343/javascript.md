# JavaScript? The F*%k is a JavaScript?

JavaScript is a weakly typed, dynamic, multi-paradigm language that is pretty much ubiquitous across all web-based software. It is so common actually, that every modern browser has a built in JavaScript REPL environment. Oh yeah, and it was built in only 1 week

## Okay, so why do I care though?

Regardless of your opinion of languages of this type and use case, JavaScript is an invaluable tool to web development. If anyone does anything relating to web technology, especially in the front-end space, JavaScript is the tool of choice every time. 

Boasting an expressive and easy to use syntax, weak typing, and support for many different types of programming styles and methodologies, it is extremely powerful for a variety of situations.

### A brief example

JavaScript can write literally every piece of your technology stack. JavaScript can be used on servers, embedded, IoT and, of course, front end. Here's a short example.

```javascript
// server.js

const express = require('express'); // A web framework for REST stuff /*1*/

const app = express; /*2*/ 
let port = process.env.PORT || 3000; /*3*/
app.set('port', port);

app.use('/hi/:me', (req, res)/*4*/ => {
    res.status(200).send(`<h1>Hey there ${me}</h1>`/*5*/);
});

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});
```

Okay, so what does this code snippet here do? 

1. This is commonly called a require statement, it is virtually the same as `import` in python or `#include` in C/C++. This is how you require external functionality defined outside of the file you're currently in

2. `const` is a keyword to define constant, immutable variables in JavaScript.

3. `let` is used to define non-constant, mutable variables in JavaScript

   #### Wait wait, pause. I have seen `var` used before, why didn't you talk about that?

   Well, young one, it's because `var` is a deprecated design pattern. Before I go into it, let (no pun intended) it be known that it is always a best practice to use `const` AS MUCH AS POSSIBLE. `const` allows for data to be much more accurate and it assures that the value won't be changed via side effects etc. Libraries like deep freeze js extend this, but I digress. That being said, we do need to use `let` sometimes, and in these cases we will always use let over var. Here's why:

   There is a very simple reason followed by others, but the biggest is: **let provides block scoping**. JavaScript for awhile was like the wild wild west. There were strange things that it did since it was made so fast and one of those was some weird scoping with the use of `var`. Var provided **function** scoping, but didn't provide an explicit restriction outside of that. This basically means that if I declare a variable outside of a function. Here's a short example:

   ```javascript
   // sample1.js
   var outputs = [];
   (function() {
       for (var i = 0; i < 5; i++) {
           outputs.push(function() { return i; });
       }  
   })();
   
   console.log(outputs.map(f => f())); // output [5, 5, 5, 5, 5]
   
   // Adapted from: https://hackernoon.com/why-you-shouldnt-use-var-anymore-f109a58b9b70
   ```

   So why does this happen? This is because of function hoisting. When functions are declared in the standard function syntax (i.e. `function foo() {}`) the JavaScript runtime does what is called "hoisting". This hoisting causes the functionality to be moved into equal scope at the top of the parent container. This is how we are able to declare functions out of order and call them without any worry, a language trait absent from older languages like C/C++.

   Here's how function hoisting makes our code weird:

   ```javascript
   // sample1.js
   var outputs = [];
   (function() {
       var i; // The var i is hoisted to the top of the function block
       for (i = 0; i < 5; i++) {
           outputs.push(function() { return i; }); // 1
       }  
   })();
   
   console.log(outputs.map(f => f())); // output [5, 5, 5, 5, 5]
   ```

   When `var i` is hoisted to the top of the function block, it allows our code to iterate i to 5 before it will push any values of i onto the callback stack because of our use of the standard function syntax on comment 1 above. i is now scoped into the blocks of the anonymous function which is actually what is pushed into the outputs array. When it is evaluated, i has already reached 5 and, as a result, when it finally gets evaluated outside of our anonymous function, it has 5 stored.

   `let` fixes our problems for us by providing much more succinct scoping and allowing things to be declared at a function level and not re scoped for each encapsulated anonymous function. Here's how it will fix the problem.

   ```javascript
   // sample2.js
   let ouptuts = []; // changing this to let didn't do anything, just for congruency
   (function() { // 1
       for (let i = 0; i < 5; i++) {
           outputs.push(function() { return i; }); // 2
       }
   })();
   
   console.log(outputs.map(f => f())); // output [0, 1, 2, 3, 4]
   ```

   This produces the output just like we expected it to. Now we see in this case the let is only declared as a member of the entire anonymous function 1 and when it gets down to function two it is using the.