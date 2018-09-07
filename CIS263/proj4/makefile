run:
	touch files && ls -alR ~/ > files
	gawk -f filesummary.awk files 
	gawk -f pwfile.awk /etc/passwd
clean:
	rm -r files
