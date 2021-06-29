no_files = 38
filename = "BERT_base_restaurant_2015"
filenames = list()
for i in range(no_files):
    filenametemp = filename + '_' + str(i) + ".txt"
    filenames.append(filenametemp)

with open(filename + '.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)


