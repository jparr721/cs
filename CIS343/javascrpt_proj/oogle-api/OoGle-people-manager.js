// The API toolkit for making REST systems easily
const express = require('express');
// A good solution for handling JSON data in routes
const bodyParser = require('body-parser');
// Node JS modules for filesystem access
const fs = require('fs');
// Our database connection
// This will be a JSON object of our programmers
// and can be accessed as if it was any other javascript
// object
const database = require('./programmers.json');

// Make an instance of our express application
const app = express();
// Specify our > 1024 port to run on
const port = 3000;

// Apply our middleware so our code can natively handle JSON easily
app.use(bodyParser.json());

// We must have our list of programmers to use
if (!fs.existsSync('./programmers.json')) {
  throw new Error('Could not find database of programmers!');
}

// Build our routes

app.get('/', (req, res) => {
  res.send('Fill me in to return ALL programmers!');
});

app.get('/:id', (req, res) => {
  const id = req.params.id;

  res.send(`Fill me in to return values with ID: ${id}`);
});

app.put('/:id', (req, res) => {
  const id = req.params.id;

  res.send(`Fill me in to update values with ID: ${id}`);
});

app.post('/', (req, res) => {
  const body = req.body; // Hold your JSON in here!

  res.send(`You sent: ${body}`);
});

// IMPLEMENT A ROUTE TO HANDLE ALL OTHER ROUTES AND RETURN AN ERROR MESSAGE

app.listen(port, () => {
  console.log(`She's alive on port ${port}`);
});
