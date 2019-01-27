#!/bin/sh

./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py 1 y" "python3.6 dummy.py"
