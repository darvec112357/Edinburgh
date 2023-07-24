import pytest
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP
from ipaddress import ip_address, ip_network
from pcap_aggr import Data, Node
from ipaddress import ip_network

testfile1 = '202011251400-78-5k.pcap'
testfile2 = '202011251400-78.pcap.gz'
testdata_aggr = {'0.0.0.0/32': 23697750, '0.0.0.0/3': 25217969, '16.0.0.0/32': 15432942, '23.0.0.0/32': 15505116, '32.0.0.0/4': 14507501, '0.0.0.0/1': 16823361, '144.0.0.0/5': 21679834, '128.0.0.0/3': 17128293, '128.0.0.0/32': 16555662, '160.0.0.0/3': 14740888, '192.0.0.0/6': 14392538, '192.0.0.0/32': 16399168, '192.0.0.0/4': 18340291, '203.62.160.0/21': 21945680, '0.0.0.0/0': 11236571}

def gotree(n):
    if n.left:
        assert int(n.ip) > int(n.left.ip)
        gotree(n.left)
    print(n)
    if n.right:
        assert int(n.ip) < int(n.right.ip)
        gotree(n.right)

def test_pcap_aggr1():
    root = None
    for pkt, _ in RawPcapReader(testfile2):
        ether = Ether(pkt)
        if not 'type' in ether.fields:
            continue
        if ether.type != 0x0800:
            continue
        ip = ether[IP]
        if root is None:
            root = Node(ip_address(ip.src), ip.len)
        else:
            root.add(ip_address(ip.src), ip.len)
    gotree(root)

def test_pcap_aggr2():
    data = Data(testfile2)
    for k, v in testdata_aggr.items():
        assert data.data[ip_network(k)] == v
