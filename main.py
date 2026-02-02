import sys
import mini_agent
from mini_agent.cli import main as cli_main

if __name__ == "__main__":
    # liujia: remove main.py from sys.argv, skip ourself.... 
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    sys.argv = ["mini-agent"] + args
    cli_main()