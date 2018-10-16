#!/usr/bin/python

from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def disableArp(router, interface):
    path = "/proc/sys/net/ipv4/conf/"+interface+"/arp_ignore"
    router.cmd('echo 8 > '+path);

def disableIcmpEcho(router):
    router.cmd('echo 1 > /proc/sys/net/ipv4/icmp_echo_ignore_all')

def prj4Net():
    info("Creating network")
    net = Mininet( controller=Controller )
    info( '*** Adding end hosts\n' )
    h1 = net.addHost( 'h1' )
    h2 = net.addHost( 'h2' )
    h3 = net.addHost( 'h3' )
    h4 = net.addHost( 'h4' )
    h5 = net.addHost( 'h5' )
    info( '*** Adding host for routers\n' )
    r1 = net.addHost( 'r1' )
    r2 = net.addHost( 'r2' )
    info( '*** Creating links\n' )
    net.addLink( r1, r2, intfName1='r1-eth0', intfName2='r2-eth0' )
    net.addLink( h1, r1, intfName1='h1-eth0', intfName2='r1-eth1' )
    net.addLink( h2, r1, intfName1='h2-eth0', intfName2='r1-eth2' )
    net.addLink( h3, r2, intfName1='h3-eth0', intfName2='r2-eth1' )
    net.addLink( h4, r2, intfName1='h4-eth0', intfName2='r2-eth2' )
    net.addLink( h5, r2, intfName1='h5-eth0', intfName2='r2-eth3' )
    info( '*** Starting network\n')
    net.start()
    info( '*** Disabling ARP on router\n' )
    disableArp( r1, 'r1-eth0' )
    disableArp( r1, 'r1-eth1' )
    disableArp( r1, 'r1-eth2' )
    disableArp( r2, 'r2-eth0' )
    disableArp( r2, 'r2-eth1' )
    disableArp( r2, 'r2-eth2' )
    disableArp( r2, 'r2-eth3' )
    info( '*** Disabling ICMP echo on router\n' )
    disableIcmpEcho( r1 )
    disableIcmpEcho( r2 )
    info( '*** Setting Addresses\n' )
    r1.setIP( '10.0.0.1', prefixLen=16, intf='r1-eth0' )
    r2.setIP( '10.0.0.2', prefixLen=16, intf='r2-eth0' )
    r1.setIP( '10.1.0.1', prefixLen=24, intf='r1-eth1' )
    r1.setIP( '10.1.1.1', prefixLen=24, intf='r1-eth2' )
    r2.setIP( '10.3.0.1', prefixLen=24, intf='r2-eth1' )
    r2.setIP( '10.3.1.1', prefixLen=24, intf='r2-eth2' )
    r2.setIP( '10.3.4.1', prefixLen=24, intf='r2-eth3' )
    h1.setIP( '10.1.0.3', prefixLen=24 )
    h2.setIP( '10.1.1.5', prefixLen=24 )
    h3.setIP( '10.3.0.32', prefixLen=24 )
    h4.setIP( '10.3.1.201', prefixLen=24 )
    h5.setIP( '10.3.4.54', prefixLen=24 )
    info( '*** Setting Default Routes\n')
    h1.setDefaultRoute( 'via 10.1.0.1')
    h2.setDefaultRoute( 'via 10.1.1.1')
    h3.setDefaultRoute( 'via 10.3.0.1')
    h4.setDefaultRoute( 'via 10.3.1.1')
    h5.setDefaultRoute( 'via 10.3.4.1')
    info( '*** Running CLI\n' )
    CLI( net )
    info( '*** Stopping network' )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    prj4Net()
