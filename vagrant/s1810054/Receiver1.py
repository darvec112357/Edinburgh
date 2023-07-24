#Rongxing Liu s1810054
import socket
import sys

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

if __name__ == "__main__":
    port=int(sys.argv[1])
    filename=sys.argv[2]
    s.bind(('localhost',port))
    f=open(filename,'wb')
    while (True):
        data,client_addr=s.recvfrom(1024)
        if(str(data)!="b'end'"):
            f.write(data)
        else:
            break
    f.close()
    s.close()
