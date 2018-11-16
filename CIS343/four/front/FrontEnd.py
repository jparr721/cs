import curses
import curses.textpad
import os
import sys
from error import Base as b


class FrontEnd(object):
    """A class representing the front end for the audio player."""
    def __init__(self, player):
        """Initialize the front end with a player instance."""
        self.player = player
        self.player.play('media/cello.wav')
        self.root_directory_files = []
        if len(sys.argv) < 2:
            raise b.CLIError('Invalid number of arguments provided')
        else:
            self.media_root = sys.argv[1]
        curses.wrapper(self.menu)

    def menu(self, args):
        """Create the menu and start the menu loop."""
        self.stdscr = curses.initscr()
        self.stdscr.border()
        self.stdscr.addstr(0, 0, 'cli-audio', curses.A_REVERSE)
        self.stdscr.addstr(5, 10, 'c - Change current song')
        self.stdscr.addstr(6, 10, 'p - Play/Pause')
        self.stdscr.addstr(7, 10, 'l - Library')
        self.stdscr.addstr(9, 10, 'ESC - Quit')
        self.update_song()
        self.stdscr.refresh()
        while True:
            c = self.stdscr.getch()
            if c == 27:
                self.quit()
            elif c == ord('p'):
                self.player.pause()
                self.update_song()
            elif c == ord('c'):
                try:
                    self.change_song()    
                    self.update_song()
                except b.CLIAudioFileException:
                    self.display_error(
                    'Could not find the desired audio file. Check your library to make sure the file is there.')
                self.stdscr.touchwin()
                self.stdscr.refresh()
                    
            elif c == ord('l'):
                self.list_directory()

    def display_error(self, error):
        """Display an error."""
        self.stdscr.addstr(15, 10, '                                        ')
        self.stdscr.addstr(15, 10, error)
                
    def update_song(self):
        """Update the song."""
        self.stdscr.addstr(15, 10, '                                        ')
        self.stdscr.addstr(
                15, 10, 'Now playing: ' + self.player.getCurrentSong())

    def play(self, song):
        """Play a song with the player."""
        try:
            self.player.play(song)
            self.update_song()
        except b.CLIAudioFileException:
            print('Invalid input file provided')

    def set_media_root(self, root):
        """Set the media root location."""
        self.media_root = root

    def get_media_root(self):
        """Return the media root location."""
        return self.media_root

    def change_song(self):
        """Change the song."""
        changeWindow = curses.newwin(5, 40, 5, 50)
        changeWindow.border()
        changeWindow.addstr(0, 0, 'What is the song name?', curses.A_REVERSE)
        self.stdscr.refresh()
        curses.echo()
        path = changeWindow.getstr(1, 1, 30)
        curses.noecho()
        self.stdscr.touchwin()
        self.stdscr.refresh()
        self.player.stop()
        self.player.play(self.media_root + '/' + path.decode(encoding='utf-8'))

    def list_directory(self):
        """
        Lists directories of current folder and displays
        only the files that are in the directory and stores
        them into the self. This is just a surbey of the songs,
        when selecting a song a different command is used
        """
        self.root_directory_files = [f for f in os.listdir(self.media_root)
                                     if os.path.isfile(
                                         os.path.join(self.media_root, f))]
        list_window = curses.newwin(
                len(self.root_directory_files) + 5, 30, 0, 90)
        list_window.border()
        list_window.addstr(0,
                           0,
                           self.media_root,
                           curses.A_REVERSE)

        self.stdscr.refresh()
        curses.noecho()
        for idx, val in enumerate(self.root_directory_files):
            list_window.addstr(idx + 1, 10, val)

        list_window.addstr(
                len(self.root_directory_files) + 3, 10, 'Press ENTER to exit')
        list_window.getstr(1, 1, 30)
        curses.noecho()
        del list_window
        self.stdscr.touchwin()
        self.stdscr.refresh()

    def quit(self):
        """Quit the front end view."""
        self.player.stop()
        exit()
