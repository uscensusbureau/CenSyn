from censyn.programs.censyn import Censyn


def command_line_start(*args, **kwargs) -> None:
    cen_process = Censyn(args, kwargs)
    if cen_process.valid_process:
        cen_process.execute()


if __name__ == '__main__':
    command_line_start()
