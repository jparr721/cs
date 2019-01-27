#!/bin/sh

if (( $# != 3 )); then
    echo "Usage: bash ./train.sh episodes batches update(y/n)"
    exit 1
fi

timestamp=$(date +%s)

for i in $(seq 1 $2)
do
  echo batch $i  
  for j in $(seq 1 $1)
  do
    echo episode $j  
    ./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py ${j} $3" "python3.6 dummy.py"
  done
  if [ "$3" == "y" ]; then
      cp model/state saved/state_batch${j}_episode${i}_${timestamp}
      python3.6 update.py $3
  fi
done
