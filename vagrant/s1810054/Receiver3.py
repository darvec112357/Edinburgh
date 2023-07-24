#Rongxing Liu s1810054
import socket
import sys
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if __name__ == "__main__":
    port=int(sys.argv[1])
    filename=sys.argv[2]
    s.bind(('127.0.0.1',port))
    f=open(filename,'wb')
    expected_seq=0
    seq=0
    while (True):
        pkt,client_addr=s.recvfrom(1027)
        data=pkt[3:1027]
        eof=pkt[2]
        seq=int.from_bytes(pkt[0:2],'big')
        if(eof==0):
            s.sendto(seq.to_bytes(2,'big'),client_addr)
            if(seq==expected_seq):
                f.write(data)
                expected_seq+=1
        elif(eof==1):
            if(seq==expected_seq):
                for i in range (10):
                    s.sendto(seq.to_bytes(2,'big'),client_addr)
                f.write(data)
                break
    f.close()
    s.close()
