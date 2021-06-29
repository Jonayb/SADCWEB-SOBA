no_files = 38  # number of files that were created by getBERT code (1 plus the highest numbered file)
filename = "BERT_base_restaurant_2015"  # the base name of the loose files
filenames = list()
for i in range(no_files):
    filenametemp = filename + '_' + str(i) + ".txt"
    filenames.append(filenametemp)

with open(filename + '.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)


