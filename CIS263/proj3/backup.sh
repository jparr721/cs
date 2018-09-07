#!/bin/bash
regex="^-."
ARGUMENTS=()
FILES=()

function handle_flags() {
	case ${ARGUMENTS[@]} in
		-l)
			echo Listing files in .backup: 
			ls -al ~/.backup | awk '{print $9}'
			;;
		-n)
			printf "There are`ls ~/.backup |wc -l` files that take up `du -ch ~/.backup | tail -1 | head -c 4` of space in .backup\n\n"
			;;
		-ln | -nl)
			printf "There are`ls ~/.backup |wc -l` files that take up `du -ch ~/.backup | tail -1 | head -c 4` of space in .backup\n\n"
			ls -al ~/.backup | awk '{print $9}'
			;;
		--help)
			printf "\n\n"
			echo "BACKUP(1)		Extended Command Manual		BACKUP(1)"
			echo "-n	Show the number of files in .backup and how much space they take up"
			echo "-l	List files currently in -l"
			echo "-h	Show help page"
			;;
		-ln--help | -nl--help | --help-ln | --help-nl)
			echo Listing files in .backup: 
			ls -al ~/.backup | awk '{print $9}'
			printf "There are`ls ~/.backup|wc -l` files that take up `du -ch ~/.backup | tail -1 | head -c 4` of space in .backup\n\n"
			printf "\n\n"
			echo "BACKUP(1)		Extended Command Manual		BACKUP(1)"
			echo "-n	Show the number of files in .backup and how much space they take up"
			echo "-l	List files currently in -l"
			echo "-h	Show help page"
			;;
		-l--help | --help-l)
			echo Listing files in .backup: 
			ls -al ~/.backup | awk '{print $9}'
			printf "\n\n"
			echo "BACKUP(1)		Extended Command Manual		BACKUP(1)"
			echo "-n	Show the number of files in .backup and how much space they take up"
			echo "-l	List files currently in -l"
			echo "-h	Show help page"
			;;
		-n--help | --help-n)
			printf "There are`ls ~/.backup|wc -l` files that take up `du -ch ~/.backup | tail -1 | head -c 4` of space in .backup\n\n"
			printf "\n\n"
			echo "BACKUP(1)		Extended Command Manual		BACKUP(1)"
			echo "-n	Show the number of files in .backup and how much space they take up"
			echo "-l	List files currently in -l"
			echo "-h	Show help page"
			;;
		*)
			echo "Invalid argument $ARGUMENTS"
		esac

}

function clean() {
	echo "Cleaning old .backup files"
	rm -rf ~/.backup
}

function backup_files() {
		clean
		echo "Making .backup..."
		mkdir ~/.backup
		cp -R ${FILES[@]} ~/.backup
}

for input in "$@"
do
	if [[ $input =~ $regex ]];
	then
		#Arguments
		ARGUMENTS+=$input
	else
		#Files
		FILES+="`pwd`/$input "
	fi
done
if [ ${#FILES[@]} -eq 0 ] && [ ${#ARGUMENTS[@]} -eq 0 ] 
then
	echo "Error, no arguments. The usage of this command is: ./backup.sh [options] targetFileList"
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
	echo ""
else
	backup_files ${FILES[@]}
fi
if [[ ${#ARGUMENTS[@]} -eq 0 ]]; then
	echo ""
else
	handle_flags ${ARGUMENTS[@]}
fi

