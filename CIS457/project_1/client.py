import socket
import sys


class Client():
    def __init__(self, port, host):
        self.host = host
        self.port = port
        self.sockfd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.PACKET_SIZE = 1024

    def run(self, filename):
        print('Requesting file: {}'.format(filename))
        self.sockfd.sendto(bytes(filename, 'UTF-8'), (self.host, self.port))

        with open(filename, 'wb') as fp:
            try:
                num, current = self.sockfd.recv(self.PACKET_SIZE)
                fp.write(current)
            except sockfd.error as err:
                print(err)

        self.sockfd.close()


if __name__ == '__main__':
    host = input('Please enter the host address: ')
    port = input('Please enter the desired port: ')
    filename = input('Enter the name of the file you want: ')
    client = Client(host, int(port))
    client.run(filename)
