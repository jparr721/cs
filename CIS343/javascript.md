# A practical guide to modern JavaScript

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
    res.status(200).send(`<h1>Hey there ${me}</h1>`);
});

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});
```

### Okay, so what does this code snippet here do?

1. This is commonly called a require statement, it is virtually the same as `import` in python. This is how you require external functionality defined outside of the file you're currently in via the use of modules (covered below).

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

   This produces the output just like we expected it to. Now we see in this case the let is only declared as a member of the entire anonymous function 1 and when it gets down to function 2 it is using the *same* reference instead of being a custom scoped reference that is not able to be mutated until the for loop exits and holds the final value. Since the for loop in our previous example shows the var being actually pulled out of the context of the for loop, it's scope broadens as well, in this case,  the let allows it to be block scoped and stay within the context it was initially intended to be placed in. This keeps the same reference when passed into the function.

4. This is called a function **callback** and these are CRITICAL to any JavaScript engineer's toolkit. Callbacks are functionality that occurs after the main purpose of the function has been completed. To understand this, we must follow one cardinal law of JavaScript: it is an **EVENT DRIVEN** language. Let that sink in and help shape your understanding of how it behaves in the wild wild web. In our referenced example above, our function sets up a route handler, then as a callback allows us to send data along the Request and Response objects.

## Event Driven, Async Architecture? Okay I get it, but how I run it?

JavaScript has the unique ability to run across many platforms and is supported on nearly any system that can run the v8 browser engine. As a result, we are capable of running it either in the browser via use of html to serve up the results, or we can just install **node js** (or deno if you're particularly savvy), a JavaScript runtime that lets you execute your JavaScript code as if it were a command line tool like any other language such as Python. 

Nodejs is important because it allows us to make and run JavaScript programs without needing to embed it into a web-page and do any extra garbage to see our results. It also compiles cleanly down to C++ code so it is high performance and great for building distributed architecture.

The nodejs REPL is almost the same as Python's REPL. It can read the files top to bottom and execute them in a script fashion. An example program looks like any old JavaScript code you've seen:

```js
// sample3.js
function say_hi(name) {
    return `Hi! ${name}`
}

console.log(say_hi('Jarred'))
```

Then you just have to run `node sample3.js` and voila, your code will run no problem!

## Every language sucks at something, what about JavaScript?

JavaScript handles a lot of things in ways that make your head scratch. In a broad sense, this is just due to how JavaScript handles things like implicit type conversion and other type handling. The most important though, is its super weird comparisons. 

If you've ever written JavaScript in any capacity, you will find that there are two ways to check for equality among values. First, we have the identity operator `===` and `!==`, and the equality operator `==` and `!=`. The identity operator works almost the same as the equality operator, but it performs no type conversion. As a common practice, it is almost never advised to use the equality operator. The equality operators can be seen as the evil twins of the identity operators. Equality operators lead to code that is weird and may not do what you want. Here's an example:

```javascript
'' == '0'           // false
0 == ''             // true
0 == '0'            // true

false == 'false'    // false
false == '0'        // true

false == undefined  // false
false == null       // false
null == undefined   // true

2 < "12" 	 		// true

' \t\r\n ' == 0     // true

[] == ![]; // -> true
```

JavaScript also has a 3 different kinds of true/false operators: `true, truthy, 1` and `false, falsy, 0`. And they are not always equal. As a common practice it is best to just use true and false values. The truthy value applies typically to the conversion of data structures like arrays down to their lowest form. For example, an empty array `[]` is considered truthy, but it is not `true`. Weird, right?

There are many best practices that can solve a lot of these issues, but they still come up and they are one of the things the JavaScript community gripes about most heavily.

## Modern JavaScript

Modern JavaScript is represented in my first code segment above, but it is largely the collection of standards brought about in ES6. This also includes new functional programming paradigms like `map`, `reduce`, and `forEach` and allows for much more readable and bulletproof code. But it also introduces some neat patterns in the form of the following:

### Modules

Modules allow functionality to be passed around in the form of inputs (as shown above). This allows for someone to export functionality from its current file and have it be exported into whatever file they want to use it in via the use of encapsulation. Here's an example:

```javascript
// calculate.js
function add(a, b) {
    return a + b;
}

module.exports = add;

// otherfunction.js
const add = require('./calculate'); //'calculate' is the function name (imports don't need the js extention)

const result = add(1, 2);

return result; // 3
```

The add function was able to be wrapped up as a module, moved around to wherever it is needed, and invoked on command. When the module is something we have imported from our local file structure, then we must use the relative path.

There is also newer syntax, but node.js doesn't support it and we still need polyfills in the browser, so we don't use it quite yet.