#!/bin/gawk -f
BEGIN {
print "directories  files  links  total  storage(bytes)"
print "================================================"
} 
{OFS="\t"}
/^-[rwxts-]{9}/{++files}
/^d[rwxts-]{9}/{++directory}
/^l[rwxts-]{9}/{++link}
{total = files+directory+link}
{bytes+=$5}
END {print directory"            "files,link"    "total"      "bytes}

