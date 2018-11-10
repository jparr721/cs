import sys
import os
import errno
from Base import Error


class Media(object):
    def __init__(self):
        self.media_root = sys.argv[1]

    def set_media_root(self, root):
        self.media_root = root

    def get_media_root(self):
        return self.media_root

    def create_playlist(self, playlist_name):
        if not os.path.exists(os.path.dirname(playlist_name)):
            try:
                os.makedirs(os.path.dirname(playlist_name))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise OSError('This directory already exists')

    def add_to_playlist(self, playlist_name, song_name):
        try:
            playlist_file_path = \
                os.path.abspath(playlist_name) + '/' + song_name
        except Error.Error:
            raise Error.Error('Playlist not found!')
        with open(os.path.abspath(song_name), "w") as f:
            with open(playlist_file_path, "w") as song:
                for line in f:
                    song.wrtie(line)
