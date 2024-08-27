import sys
import subprocess
from utils import *

# parser = get_arg_parser()
# args = parser.parse_args()

# print(args)
print(f'Executing evaluation for the following configuration: {sys.argv[1:]}')
for task in TASKS:
    sysargs = sys.argv[1:] + ['--task', task]
    print(f'Evaluating {task}')
    cmd = ['python', f'test-{task}.py', ] + sysargs
    process = subprocess.Popen(cmd)
    process.wait()
    print('Done!\n______________________________________________\n')

print("Done")