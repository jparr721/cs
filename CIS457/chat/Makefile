all:
	g++-7 -std=c++17 -o client ChatClient.cc Succ.cc Crypto.cc -g -lssl -lcrypto -lpthread
	g++-7 -std=c++17 -o server ChatServer.cc Succ.cc Crypto.cc -g -lssl -lcrypto -lpthread

t:
	g++-7 -std=c++17 -o test Test.cc Crypto.cc -g -lssl -lcrypto -lpthread
