from pyspark import SparkContext
import sys
from itertools import combinations
from itertools import product
import time

start_time = time.time()

# make output file data
candidates = []
frequents = []
support = int(sys.argv[2])

sc = SparkContext.getOrCreate()

# read csv file
rddFromFile = sc.textFile(str(sys.argv[3]))
header = rddFromFile.first()
rddFromFile = rddFromFile.filter(lambda row: row != header) # filter out the header

# make basket by case number
if(sys.argv[1] == '1'):
    rddSplit = rddFromFile.distinct().map(lambda f:f.split(","))
    rdd = rddSplit.groupByKey().mapValues(set).map(lambda g: g[1]).collect()

else:
    rddSplit = rddFromFile.distinct().map(lambda f: (f.split(",")[1], f.split(",")[0]))
    rdd = rddSplit.groupByKey().mapValues(set).map(lambda g: g[1]).collect()


# count singles
singletons = rddSplit.map(lambda g: g[1]).countByValue().items()
candidate_singles = []
frequent_singles = []

for single in singletons:
    candidate_singles.append(single[0])
    if(single[1] >= support):
        frequent_singles.append(single[0])

    candidate_singles = sorted(candidate_singles)
    frequent_singles = sorted(frequent_singles)
    
candidates.append(' ' + str(candidate_singles).replace(",", "),").replace(" '", "('") + ' ')
frequents.append(' ' + str(frequent_singles).replace(",", "),").replace(" '", "('") + ' ')

frequent = True
comb_val = 2

# start A-Priori
while(frequent):
    combs = list(combinations(frequent_singles, comb_val))

    data = []
    
    for basket in rdd:
        for comb in combs:
            if(set(comb).issubset(basket)):
                data.append(comb)
    
    mapping = sc.parallelize(data)
    reduced = mapping.countByValue().items()
    
    candidate_items = []
    frequent_items = []
    
    for item in reduced:
        candidate_items.append(item[0])
        if(item[1] >= support):
            frequent_items.append(item[0])
    
    candidate_items = sorted(candidate_items)
    frequent_items = sorted(frequent_items)
    candidates.append(candidate_items)
    frequents.append(frequent_items)

    comb_val = comb_val + 1

    if(len(frequent_items) == 0):
        frequent = False

# write to file
with open(str(sys.argv[4]), 'w') as f:
    f.write("Candidates:\n")
    for candidate in candidates:
        f.write(str(candidate).replace('[','(').replace(']',')')[1:-1])
        f.write('\n\n')
    
    f.write("Frequent Itemsets:\n")
    for frequent in frequents:
        f.write(str(frequent).replace('[','(').replace(']',')')[1:-1])
        f.write('\n\n')

duration = time.time() - start_time

print("Duration: ", duration)
