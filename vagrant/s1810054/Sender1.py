#Rongxing Liu s1810054
import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if __name__ == "__main__":
    host=sys.argv[1]
    port=int(sys.argv[2])
    filename=sys.argv[3]
    client_addr=(host,port)
    f=open(filename,'rb')
    while(True):
        data=f.read(1024)
        if(str(data)!="b''"):
            s.sendto(data,client_addr)
        else:
            s.sendto('end'.encode('utf-8'),client_addr)
            break
    print("Finished!")
    s.close()
