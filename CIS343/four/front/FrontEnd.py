import curses
import curses.textpad
import os


class FrontEnd:

    def __init__(self, player):
        self.player = player
        self.player.play('media/cello.wav')
        curses.wrapper(self.menu)
        self.root_directory_files = []

    def menu(self, args):
        self.stdscr = curses.initscr()
        self.stdscr.border()
        self.stdscr.addstr(0, 0, "cli-audio", curses.A_REVERSE)
        self.stdscr.addstr(5, 10, "c - Change current song")
        self.stdscr.addstr(6, 10, "p - Play/Pause")
        self.stdscr.addstr(7, 10, "l - Library")
        self.stdscr.addstr(9, 10, "ESC - Quit")
        self.update_song()
        self.stdscr.refresh()
        while True:
            c = self.stdscr.getch()
            if c == 27:
                self.quit()
            elif c == ord('p'):
                self.player.pause()
            elif c == ord('c'):
                self.change_song()
                self.update_song()
                self.stdscr.touchwin()
                self.stdscr.refresh()
            elif c == ord('l'):
                self.list_directory()

    def update_song(self):
        self.stdscr.addstr(15, 10, "                                        ")
        self.stdscr.addstr(
                15, 10, "Now playing: " + self.player.getCurrentSong())

    def play(self, song):
        self.player.play(song)

    def change_song(self):
        changeWindow = curses.newwin(5, 40, 5, 50)
        changeWindow.border()
        changeWindow.addstr(0, 0, "What is the file path?", curses.A_REVERSE)
        self.stdscr.refresh()
        curses.echo()
        path = changeWindow.getstr(1, 1, 30)
        curses.noecho()
        del changeWindow
        self.stdscr.touchwin()
        self.stdscr.refresh()
        self.player.stop()
        self.player.play(path.decode(encoding="utf-8"))

    def list_directory(self):
        '''
        Lists directories of current folder and displays
        only the files that are in the directory and stores
        them into the self. This is just a surbey of the songs,
        when selecting a song a different command is used
        '''
        self.root_directory_files = [f for f in os.listdir('./')
                                     if os.path.isfile(os.path.join('./', f))]
        list_window = curses.newwin(5, 200, 5, 200)
        list_window.border()
        list_window.addstr(0,
                           0,
                           'Songs in current directory',
                           curses.A_REVERSE)
        for idx, val in enumerate(self.root_directory_files):
            list_window.addstr(idx + 1, 10, val)
        self.stdscr.refresh()
        curses.noecho()
        del list_window
        self.stdscr.touchwin()
        self.stdscr.refresh()

    def quit(self):
        self.player.stop()
        exit()
