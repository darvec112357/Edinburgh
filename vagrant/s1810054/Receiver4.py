#Rongxing Liu s1810054
import socket
import sys
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if __name__ == "__main__":
    port=int(sys.argv[1])
    filename=sys.argv[2]
    N=int(sys.argv[3])
    s.bind(('127.0.0.1',port))
    f=open(filename,'wb')
    expected_seq=0
    seq=0
    buffer={}
    while (True):
        pkt,client_addr=s.recvfrom(1027)
        data=pkt[3:1027]
        eof=pkt[2]
        seq=int.from_bytes(pkt[0:2],'big')
        if(eof==0):
            if(N==1):
                s.sendto(seq.to_bytes(2,'big'),client_addr)
                if(seq==expected_seq):
                    #if seq matches the expected_seq, we can write into our files
                    f.write(data)
                    expected_seq+=1
                    while(expected_seq in buffer.keys()):
                        #write all the available buffered data
                        f.write(buffer[expected_seq])
                        del buffer[expected_seq]
                        expected_seq+=1
                elif(seq>expected_seq):
                    #otherwise we can buffer the data
                    buffer[seq]=data
            else:
                if(expected_seq-N<=seq and seq<expected_seq+N):
                    s.sendto(seq.to_bytes(2,'big'),client_addr)
                    if(expected_seq<=seq and seq<expected_seq+N):
                        if(seq==expected_seq):
                            f.write(data)
                            expected_seq+=1
                            while(expected_seq in buffer.keys()):
                                f.write(buffer[expected_seq])
                                del buffer[expected_seq]
                                expected_seq+=1
                        elif(seq>expected_seq):
                            buffer[seq]=data
        elif(eof==1):
            if(seq==expected_seq):
                for i in range (10):
                    s.sendto(seq.to_bytes(2,'big'),client_addr)
                f.write(data)
                break
    f.close()
    s.close()
