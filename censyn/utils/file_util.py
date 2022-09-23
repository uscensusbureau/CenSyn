

def append_to_file_name(file_name: str, file_append: str = ""):
    """
    Append string to name of file.

    :param: file_name: File name with extension.
    :param: file_append: String to append to name.
    :return: Modified file name.
    """
    if not file_append:
        return file_name
    parts = file_name.rsplit('.', 1)
    fileout = parts[0] + file_append + '.' + parts[1]
    return fileout
