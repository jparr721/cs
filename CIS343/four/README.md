# cli-audio

This is an incredibly simplistic CLI audio player.  The purpose is not to be a complete solution, but a starter project for students to enhance.  Written for a CIS 343 class at GVSU.

Note that in its current state it only plays .wav files.

## Requirements
Requires pyaudio and curses.  curses is usually built-in.  pyaudio easily installed via pip for Windows folks.  Better methods are available for OSX and Linux users.  See https://people.csail.mit.edu/hubert/pyaudio/ for details.

```
python -m pip install pyaudio
```
Run it with:

```
python cli-audio <media_dir>
```

Via the pyaudio website.
