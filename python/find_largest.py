import sys
file_name = sys.argv[1]
a = [float(line.rstrip()) for line in open(file_name,'r')]
print a
print max(a)
print a.index(max(a))
