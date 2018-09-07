#!/bin/bash
if [ -d $1 ] 
then
	file=$(ls $1)
	for f in $file
	do
		if [ -f $f ]
		then
			echo "$f: is a file"
		elif [ -d $f ]
		then
			echo "$f: is a directory"
		fi
	done
	
fi
if [ -f $1 ]
then
	echo "$1: an ordinary file"
fi
