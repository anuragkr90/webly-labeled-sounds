import sys
f = open(sys.argv[1], 'r')    # pass an appropriate path of the required file
lines = f.readlines()
#lines[53-1] = "    model : 'snapshot/virat_iter_" + sys.argv[2] + ".caffemodel'\n"    # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
#lines[4-1] = "exp=" + sys.argv[2] + "\n"    # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
#lines[5-1] = "exp1=" + sys.argv[2]+ "_percent_embedding_architecture \n"    # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
lines[7-1] =  "lr=" + sys.argv[2] + "\n"
f.close()   # close the file and reopen in write mode to enable writing to file; you can also open in append mode and use "seek", but you will have some unwanted old data if the new data is shorter in length.

f = open("runall_embedding_only_40_240_40_percent_10_percent_"+ sys.argv[2] + ".sh", 'w')
f.writelines(lines)
# do the remaining operations on the file
f.close()
