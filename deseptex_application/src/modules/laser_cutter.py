import socket
import numpy as np



class laser_cutter_t:
    def __init__(self, ip, udp_out_port, udp_in_port, M):
        self.ip = ip
        self.udp_out_port = udp_out_port
        self.udp_in_port = udp_in_port
        self.M = M
    
    
    def start_cutter(self):
        MESSAGE = "START"

        outSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        inSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        inSock.bind((self.ip, self.udp_in_port))

        outSock.sendto((MESSAGE).encode(), (self.ip, self.udp_out_port))
        data, addr = inSock.recvfrom(1024)
        print("start cutter: ", data, addr)
    
    
    def prepare_svg(self, contours):
        with open("./svg/test.svg", "w+") as f:
            f.write(f'<svg width="{1400}mm" height="{900}mm" xmlns="http://www.w3.org/2000/svg">')

            f.write('<path d="M0 0 1400 0 1400 900 0 900" fill="none" style="stroke:blue" />') # for some reason it is necessary to draw a rectangle such that the svg aligns with cutter software

            for points in contours:
                f.write('<path d="M')
                for i in range(len(points)):
                    
                    # get camera point
                    x_cam, y_cam = points[i]
                    cam_point = np.array([x_cam, y_cam, 1])
                    
                    # convert to machine point
                    machine_point = np.matmul(self.M, cam_point)
                    x = 1400 - (machine_point[0]/machine_point[2])
                    y = machine_point[1]/machine_point[2]

                    # write point to svg
                    f.write(f"{x} {y} ")
                f.write('" fill="none" style="stroke:red" />')
            f.write("</svg>")
    
    
    def send_svg_to_cutter(self):
        # MESSAGE = "LOADFILE:C:\\Users\\techniphys\\Documents\\DeSepTex - Photos\\test.svg"
        MESSAGE = "LOADFILE:../../svg/test.svg"

        outSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        inSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        inSock.bind((self.ip, self.udp_in_port))

        outSock.sendto((MESSAGE).encode(), (self.ip, self.udp_out_port))
        data, addr = inSock.recvfrom(1024)
        print("send svg to cutter: ", data, addr)