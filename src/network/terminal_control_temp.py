import sys
import select
try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

def enable_single_key_mode() -> tuple[object | None, object | None]:
    """Put terminal in cbreak mode so single key presses are readable."""
    if not sys.stdin.isatty() or termios is None:
        return None, None

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, previous

def disable_single_key_mode(fd: object | None, previous: object | None) -> None:
    """Restore terminal mode after single-key capture usage."""
    if fd is None or previous is None or termios is None:
        return
    termios.tcsetattr(fd, termios.TCSADRAIN, previous)

def space_pressed() -> bool:
    """Return True when a space key is waiting on stdin."""
    if not sys.stdin.isatty():
        return False

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False

    return sys.stdin.read(1) == " "
