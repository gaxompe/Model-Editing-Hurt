import sys
import subprocess
from utils import *

# parser = get_arg_parser()
args = parser.parse_args()

print(args)
cmd = ['python', f'test-{args.task}.py', ] + sys.argv[1:]

process = subprocess.Popen(cmd)


process.wait()
print("Done")