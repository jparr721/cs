import socket
import sys


class Client():
    def __init__(self, port, host):
        self.host = host
        self.port = port
        self.sockfd = socket.socket(socket.AF_INET, socket.DGRAM)
        self.PACKET_SIZE = 1024

    def run(self, filename):
        print('Requesting file: {}'.format(filename))
        self.socket.sendto(filename.encode(), (self.host, self.port))

        with open(filename, 'wb') as fp:
            try:
                num, current = self.sockfd.recv(self.PACKET_SIZE)
                fp.write(current)
            except socket.error as err:
                print(err)

        self.sockfd.close()


if __name__ == '__main__':
    host = ''
    port = 0
    filename = input('Enter the name of the file you want: ')
