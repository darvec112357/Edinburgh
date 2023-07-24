#Rongxing Liu s1810054
import socket
import sys
import time
import threading
import math

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def thread_function(send_base,addr,file):
    #The thread is used to receive ack, if the sender successfully
    #receive the ack, increase the base by 1; otherwise go back
    global send_base_copy
    global goback
    global finish
    global t
    goback=False
    finish=False
    try:
        s.settimeout(time_out/1000)
        msg=int.from_bytes(s.recv(2),'big')
        if(msg==send_base):
            send_base_copy=send_base+1
            t=0
            if(msg==math.ceil(len(file)/1024)-1):
                #Have received all the acks, the sender can close now
                finish=True
    except socket.timeout:
        goback=True

def udt_send(addr,seq,file,start):
    global t
    if(start+1024>=len(file)):
        pkt=seq.to_bytes(2,'big')+bytes([1])+file[start:]
        t+=1
    else:
        pkt=seq.to_bytes(2,'big')+bytes([0])+file[start:start+1024]
    s.sendto(pkt,addr)

if __name__ == "__main__":
    start_time=time.perf_counter()
    host=sys.argv[1]
    port=int(sys.argv[2])
    filename=sys.argv[3]
    time_out=int(sys.argv[4])
    N=int(sys.argv[5])
    addr=(host,port)
    f=open(filename,'rb')
    file=f.read()
    send_base=0
    seq=0
    start=0
    send_base_copy=0
    goback=False
    finish=False
    t=0
    #In the case of large windowsize, the receiver will close earlier.
    #If the sender sends too many eof yet not receiving ACK, the sender will know
    #that receiver has received all the packets and closed, hence the sender will also terminate.
    while(not finish and t<10):
        send_base=send_base_copy
        if(goback):
            #This means we have a timeout, change the seq back to the base,
            #and also change the index of 'start'
            seq=send_base
            start=seq*1024
        #Start a thread to receive ack
        ack_thread=threading.Thread(target=thread_function,
                                    args=(send_base,addr,file,))
        ack_thread.start()
        #Meanwhile, send next packets
        while(seq<send_base+N and start<len(file)):
            #Make sure that the seq is within the window size
            udt_send(addr,seq,file,start)
            start+=1024
            seq+=1
        ack_thread.join()
    end_time=time.perf_counter()
    time_taken=end_time-start_time
    throughput=len(file)/(time_taken*1024)
    print(throughput)
    f.close()
    s.close()
