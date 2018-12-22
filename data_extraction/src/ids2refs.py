import sys
import os

refs = {}
#  print('sys.stdin: ', sys.stdin)
#  print('sys.argv[1]: ', sys.argv[1])

for line in sys.stdin:
    line = line.rstrip()
    #  print(line)
    els = line.split("\t")
    hashstr = els[0]
    response = els[6]
    refs[hashstr] = response

#  print('refs: ', len(refs))
#  total = 0
#  missing_count = 0
with open(sys.argv[1]) as f:
    for line in f:
        line = line.rstrip()
        els = line.split("\t")
        sys.stdout.write(els[0])
        for i in range(1, len(els)):
            p = els[i].split('|')
            score = p[0]
            hashstr = p[1]
            if hashstr in refs.keys():
                sys.stdout.write("\t" + score + "|" + refs[hashstr])
            else:
                print("WARNING: missing ref, automatic eval scores may differ: [%s]" % hashstr, file=sys.stderr)
        sys.stdout.write("\n")

#  print('total: %d' % total)
#  print('missing_count: %d' % missing_count)
