from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP
from ipaddress import ip_address, ip_network
import sys
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, ip, plen):
        self.bytes = plen
        self.left = None
        self.right = None
        self.ip = ip
    def add(self, ip, plen):
        m=int(self.ip)
        n=int(ip)
        if(n<m):
            if(self.left==None):
                self.left=Node(ip,plen)
            else:
                self.left.add(ip,plen)
        elif(n==m):
            self.bytes+=plen
        else:
            if(self.right==None):
                self.right=Node(ip,plen)
            else:
                self.right.add(ip,plen)
        
    def data(self, data):
        if self.left:
            self.left.data(data)
        if self.bytes > 0:
            data[ip_network(self.ip)] = self.bytes
        if self.right:
            self.right.data(data)
            
    @staticmethod 
    def supernet(ip1, ip2):
        # arguments are either IPv4Address or IPv4Network
        na1 = ip_network(ip1).network_address
        na2 = ip_network(ip2).network_address
        netmask=0
        get_bin = lambda x: format(x, 'b').zfill(32)
        n1=get_bin(int(na1))
        n2=get_bin(int(na2))
        while(netmask<32 and n1[netmask]==n2[netmask]):
            netmask+=1
        return ip_network('{}/{}'.format(na1, netmask), strict=False)

##    def aggr(self, byte_thresh):
##        leftleaf=(self.left!=None) and (self.left.left==None) and (self.left.right==None)
##        rightleaf=(self.right!=None) and (self.right.left==None) and (self.right.right==None)
##        if(self.left!=None and (not leftleaf)):
##            self.left.aggr(byte_thresh)
##        if(self.right!=None and (not rightleaf)):   
##            self.right.aggr(byte_thresh)
##        if(self.left!=None and self.left.bytes<byte_thresh):
##            self.bytes+=self.left.bytes
##            self.ip=self.supernet(self.ip,self.left.ip)
##            if(self.left.left!=None or self.left.right!=None):
##                self.left.bytes=0
##            else:
##                self.left=None
##        if(self.right!=None and self.right.bytes<byte_thresh):
##            self.bytes+=self.right.bytes
##            self.ip=self.supernet(self.ip,self.right.ip)
##            if(self.right.left!=None or self.right.right!=None):
##                self.right.bytes=0
##            else:
##                self.right=None

    def aggr(self,byte_thresh):
        leftleaf=self.left and (self.left.left==None) and (self.left.right==None)
        rightleaf=self.right and (self.right.left==None) and (self.right.right==None)
        if(self.left):
            self.left.aggr(byte_thresh)
            if(self.left.bytes<byte_thresh):
                self.bytes+=self.left.bytes
                self.ip=self.supernet(self.ip,self.left.ip)
                self.left.bytes=0
                if(leftleaf):
                    self.left=None
        if(self.right):
            self.right.aggr(byte_thresh)
            if(self.right.bytes<byte_thresh):
                self.bytes+=self.right.bytes
                self.ip=self.supernet(self.ip,self.right.ip)
                self.right.bytes=0
                if(rightleaf):
                    self.right=None
        
class Data(object):
    def __init__(self, data):
        self.tot_bytes = 0
        self.data = {}
        self.aggr_ratio = 0.05
        root = None
        cnt = 0
        for pkt, metadata in RawPcapReader(data):
            ether = Ether(pkt)
            if not 'type' in ether.fields:
                continue
            if ether.type != 0x0800:
                continue
            ip = ether[IP]
            self.tot_bytes += ip.len
            if root is None:
                root = Node(ip_address(ip.src), ip.len)
            else:
                root.add(ip_address(ip.src), ip.len)
            cnt += 1
        root.aggr(self.tot_bytes * self.aggr_ratio)
        root.data(self.data)
    def Plot(self):
        data = {k: v/1000 for k, v in self.data.items()}
        plt.rcParams['font.size'] = 8
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(which='major', axis='y')
        ax.tick_params(axis='both', which='major')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([str(l) for l in data.keys()], rotation=45,
            rotation_mode='default', horizontalalignment='right')
        ax.set_ylabel('Total bytes [KB]')
        ax.bar(ax.get_xticks(), data.values(), zorder=2)
        ax.set_title('IPv4 sources sending {} % ({}KB) or more traffic.'.format(
            self.aggr_ratio * 100, self.tot_bytes * self.aggr_ratio / 1000))
        plt.savefig(sys.argv[1] + '.aggr.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    d = Data(sys.argv[1])
    d.Plot()
