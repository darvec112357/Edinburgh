from ipaddress import ip_address,ip_network
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet6 import IPv6
from scapy.layers.inet import IP, TCP, UDP
import sys


def supernet(ip1, ip2):
        # arguments are either IPv4Address or IPv4Network
        na1 = ip_network(ip1).network_address
        na2 = ip_network(ip2).network_address
        netmask=0
        i=31
        get_bin = lambda x: format(x, 'b').zfill(32)
        n1=get_bin(int(na1))
        n2=get_bin(int(na2))
        while(i>=0 and n1[i]==n2[i]):
            netmask+=1
            i-=1
        common=(n1>>(32-netmask))<<(32-netmask)
        addr=[(common & (0xFF << (8*n))) >> 8*n for n in (3, 2, 1, 0)]
        s=str(addr[0])+'.'+str(addr[1])+'.'+str(addr[2])+'.'+str(addr[3])
        na1=ip_address(s)
        return ip_network('{}/{}'.format(na1, netmask), strict=False)

get_bin = lambda x: format(x, 'b').zfill(32)
ip1=ip_address('208.0.0.0')
ip2=ip_address('208.0.0.0')
a=get_bin(123)
a[27]='0'
print(a[27:30])


