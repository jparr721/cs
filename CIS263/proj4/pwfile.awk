#!/bin/gawk -f
BEGIN {FS=":"
print "USER    PASS    UID    GID   NAME    HOME    SHELL"
print "================================================================="	
}
{OFS="  "}
{
}
{
print $1"  "$2"  "$3"  "$4"  "$5"  "$6"  "$7
}
END {
print "Next available uid: 65535"
print "Next available gid: 65535"
}
