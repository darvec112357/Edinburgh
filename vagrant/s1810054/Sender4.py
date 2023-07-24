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
    global acks
    global lock
    global finish
    with lock:
        while(True):
            try:
                #If nothing is received within 0.5s, sender will know that
                #receiver has terminated, and thus also terminate.
                s.settimeout(0.5)
                msg=int.from_bytes(s.recv(2),'big')
                if(msg in [i for i in range (send_base,send_base+N)]):
                    if(not acks[msg]):
                        acks[msg]=True
                        while(send_base<math.ceil(len(file)/1024) and acks[send_base]):
                            send_base+=1
                        send_base_copy=send_base
            except socket.timeout:
                finish=True
                break

def thread_resend(seq,addr,file,timeout):
    #This thread is created for each seq, used for
    #timer and retransmission purpose.
    global acks
    global finish
    while(not finish):
        #Start the timer
        time.sleep(timeout/1000)
        #Check whether ack of seq has been received
        if(acks[seq]):
            break
        udt_send(addr,seq,file)


def udt_send(addr,seq,file):
    #print(seq)
    n=seq*1024
    if(n+1024>=len(file)):
        pkt=seq.to_bytes(2,'big')+bytes([1])+file[n:]
    else:
        pkt=seq.to_bytes(2,'big')+bytes([0])+file[n:n+1024]
    s.sendto(pkt,addr)

if __name__ == "__main__":
    start_time=time.perf_counter()
    host=sys.argv[1]
    port=int(sys.argv[2])
    filename=sys.argv[3]
    timeout=int(sys.argv[4])
    N=int(sys.argv[5])
    addr=(host,port)
    f=open(filename,'rb')
    file=f.read()
    acks=[False for i in range (math.ceil(len(file)/1024))]
    send_base=0
    send_base_copy=0
    seq=0
    seq_copy=0
    finish=False
    lock=threading.Lock()

    #Start a thread to process ACK received
    ack_thread=threading.Thread(target=thread_function,
                                args=(send_base,addr,file,))
    ack_thread.start()
    while(1024*seq<len(file)):
        send_base=send_base_copy
        if(seq<send_base+N):
            #Create a thread for each seq for resend purpose,
            #while limit the seq within the windowsize to prevent from
            #having too many threads
            thread=threading.Thread(target=thread_resend,args=(seq,addr,file,timeout,))
            thread.start()
            seq+=1
        if(seq_copy<send_base+N and 1024*seq_copy<len(file)):
            #Make sure that the seq is within the window size
            udt_send(addr,seq_copy,file)
            seq_copy+=1
    ack_thread.join()
    thread.join()
    end_time=time.perf_counter()
    time_taken=end_time-start_time
    throughput=len(file)/(time_taken*1024)
    print(throughput)
