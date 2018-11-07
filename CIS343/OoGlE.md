# Welcome to OoGlE

Yes, that's right! You now work for the esteemed and luxurious company, OoGlE. Creators of such products like O-mail, OoGlE Thrive, and that one program that watches you sleep!

To help with placement on teams, the engineers have devised a test to determine how you handle real world problems. Everyone's test is unique so there won't be any cheating here! Your task, *insert name here*, is to use JavaScript to build a system to manage our employees. Here at OoGlE, we have a lot of ~~slaves~~ team members and we need a way to keep track of them for purposes you're not allowed to know! This is very exciting and will help up fulfill our mission of being hopefully not evil!

## Project Description

You will be given a large JSON file full of data for users. JSON is the Javascript Object Notation and it can be interacted with almost the exact same as a normal JavaScript object. The data will be represented in the following format:

```json
{
    "firstName": "jeff",
    "lastName": "burt",
    "homeAddress": "1234 Mayberry Lane",
    "SID": "48764-938102-87462",
    "goodSlave": "true",
    "beatingsToDate": 127,
    "family": {
        "wife": {},
        "husband": {
            "firstName": "Torfelvin",
            "lastName": "burt",
            "homeAddress": "1234 Mayberry Lane",
            "SID": "NS",
            "goodSlave": "NS",
            "beatingsToDate": -1
        },
        "children": {
            "Designation": "Jeff Jr.",
       		"Designation": "Turtle Person",
        	"Designation": "REDACTED"
        }
    }
}
```

With this information we will keep careful track of our employees via a database system (for this project we'll just load the file into memory). The goal is to design a RESTful API in JavaScript (or TypeScript) to handle the storage and retrieval of these people who are definitely not slave programmers. The layout of the routes is as follows:

```
GET / 		Returns ALL slave programmers
GET /:id 	Returns ALL slave programmers by ID
PUT /:id 	Updates a particular slave programmer (or more than one) by ID
POST /		Creates a new slave programmer into our legion
```

