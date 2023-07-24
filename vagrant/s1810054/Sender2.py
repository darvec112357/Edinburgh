#Rongxing Liu s1810054
import socket
import sys
import time
import threading

def udt_send(addr,seq,file):
    global process
    global start
    if(start+1024>=len(file)):
        pkt=seq.to_bytes(2,'big')+bytes([1])+file[start:]
        process=False
    else:
        pkt=seq.to_bytes(2,'big')+bytes([0])+file[start:start+1024]
    s.sendto(pkt,addr)

if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    retransmission=0
    process=True
    start=0
    start_time=time.perf_counter()
    host=sys.argv[1]
    port=int(sys.argv[2])
    filename=sys.argv[3]
    time_out=int(sys.argv[4])
    addr=(host,port)
    f=open(filename,'rb')
    file=f.read()
    seq=0
    while(process):
        udt_send(addr,seq,file)
        #Stop and wait until the ack has been received.
        while(True):
            try:
                s.settimeout(time_out/1000)
                msg=int.from_bytes(s.recv(2),'big')
                if(msg==seq):
                    #If received, update the seq and start to transmit next packet
                    seq+=1
                    start+=1024
                    break
            except socket.timeout:
                udt_send(addr,seq,file)
                retransmission+=1
    end_time=time.perf_counter()
    time_taken=end_time-start_time
    throughput=len(file)/(1024*time_taken)
    print(retransmission,throughput)
    f.close()
    s.close()
