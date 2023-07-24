import socket
import sys

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        
if __name__ == "__main__":
    port=sys.argv[1]
    filename=sys.argv[2]
    s.bind(("localhost",port))
    count=0
    while (True):
        data, client_addr = s.recv(1024)
        if(count==0):
            f=open(data,'wb')
        if(str(data)!="b'end'"):
            f.write(data)
        else:
            break
        count+=1
    f.close()
    s.close()
