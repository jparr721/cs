#!/bin/gawk -f
BEGIN {
	ORS = ""
	discarded = 0
	gtotal = 0
	goodrecord = 0
	nrecord = 0
	}
NR == 1 {
	nfields = NF
	}
	{
	nrecord += 1
	if ($0 ~ /[^0-9. \t]/ || NF < nfields)
	{
		print "\nRecord " NR " skipped:\n\t"
		print $0 "\n"
		discarded += 1
		next
	}
	else
	{
		goodrecord += 1
		for (count = 1; count <= nfields; count++)
		{
			printf "%8.2f", $count > "caverage.out"
			sum[count] += $count
			gtotal += $count
		}
		print "\n" > "caverage.out"
	}
	}
END	{
	print "\n"

	for (count = 1; count <= nfields; count++)
	{
		print "======" > "caverage.out"
	}
	print "\n" > "caverage.out"
	for (count = 1; count <= nfields; count++)
	{
		printf "8.2f"
