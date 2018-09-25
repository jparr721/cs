import socket
import sys
import os


class Server():
    def __init__(self, port):
        self.host = '127.0.0.1'
        self.port = port
        self.sockfd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sockfd.bind(self.host, self.port)
        self.WINDOW_SIZE = 5
        self.PACKET_SIZE = 1024

    def run(self):
        print('Server running\n')
        while True:
            file_data, client = self.sockfd.recvfrom(self.PACKET_SIZE)
            filename = file_data.decode()

            if os.path.isfile(filename):
                fp = open(filename, "rb")

                packets = self.make_packets(fp)
                fp.close()

                num_packets = len(packets)
                next_packet = 0
                window_start = 0
                window = self.WINDOW_SIZE
                print('Packets to be sent: {}'.format(num_packets))

                while window_start < num_packets:
                    while (next_packet < window_start + window
                            and next_packet < num_packets):
                        self.sock.sendto(packets[next_packet], client)
                        next_packet += 1

                    window_start += 1
                    window = min(self.WINDOW_SIZE, num_packets - window_start)
            else:
                print('Could not find file\n')

    def make_packets(self, fp):
        data = fp.read(1020)
        packets = []
        packet_number = 0

        while data:
            bytes = data.to_bytes(4, byteorder='little', signed=True)
            packets.append(bytes + data)
            packet_number += 1

        return packets


if __name__ == '__main__':
    port = 0
    if len(sys.argv) != 2:
        print("usage: python server.py port")
        sys.exit()
    else:
        port = int(sys.argv[1])

    server = Server(port)
    server.run()
