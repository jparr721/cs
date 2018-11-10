class Error(Exception):
    pass


class CLIError(Error):
    pass


class CLIAudioFileException(CLIError):
    pass


class CLIAudioScreenSizeException(CLIError):
    pass
