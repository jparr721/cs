import sys
import os
import errno
from Base import Error


class Media(object):
    """A class to store media information."""
    def __init__(self):
        """Media initialization function."""
        self.media_root = sys.argv[1]

    def set_media_root(self, root):
        """Set the root for the media."""
        self.media_root = root

    def get_media_root(self):
        """Return the root for the media."""
        return self.media_root

    def create_playlist(self, playlist_name):
        """Create a playlist with a name."""
        if not os.path.exists(os.path.dirname(playlist_name)):
            try:
                os.makedirs(os.path.dirname(playlist_name))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise OSError('This directory already exists')

    def add_to_playlist(self, playlist_name, song_name):
        """Add a song to a playlist."""
        try:
            playlist_file_path = \
                os.path.abspath(playlist_name) + '/' + song_name
        except Error.Error:
            raise Error.Error('Playlist not found!')
        with open(os.path.abspath(song_name), "w") as f:
            with open(playlist_file_path, "w") as song:
                for line in f:
                    song.wrtie(line)
