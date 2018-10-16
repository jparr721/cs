import socket
import os.path
import sys
import time


class Server():
    def __init__(self, port):
        self.host = '127.0.0.1'
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket. SOCK_DGRAM)
        self.server = (self.host, self.port)
        self.sock.bind(self.server)
        self.window_size = 5

    def listen(self):
        print('Listening on: ' + self.host + ':' + str(self.port))
        while True:
            # Get data
            payload, client_host = self.sock.recvfrom(1024)
            filename = payload.decode()

            # Open file
            if self.file_exists(filename):
                f = open(filename, 'rb')

                # Add all packets and number them
                packets = []
                num = 0
                file_contents = f.read(1020)
                while file_contents:
                    packets.append(self.make_packet(num, file_contents))
                    num += 1
                    file_contents = f.read(1020)
                f.close()

                num_packets = len(packets)
                print('Number packets: ', num_packets)
                next_frame = 0
                base = 0
                window = self.set_window(num_packets, base)

                # Send the packet
                while base < num_packets:
                    # Send all packets within the window
                    while next_frame < base + window and next_frame < num_packets:
                        print('Sending packet ', next_frame)
                        self.send_data_to_socket(packets[next_frame], client_host)
                        next_frame += 1
                        base += 1

                print("\nAwaiting next request...")

            else:
                print('File does not exist.')

    def send_data_to_socket(self, payload, host):
        self.sock.sendto(payload, host)

    def file_exists(self, filename):
        return os.path.isfile(filename)

    def set_window(self, num_packets, base):
        return min(self.window_size, num_packets - base)

    def make_packet(self, acknum, data=b''):
        ackbytes = acknum.to_bytes(4, byteorder='little', signed=True)
        return ackbytes + data


if __name__ == '__main__':

    port = sys.argv[1]

    server = Server(int(port))
    server.listen()


