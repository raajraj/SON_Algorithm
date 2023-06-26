from pyspark import SparkContext
import sys
import itertools
from itertools import combinations
from itertools import product
import time
import math
import csv
import functools
from functools import reduce
from operator import add

filter = int(sys.argv[1])
support = int(sys.argv[2])

candidates = []
frequents = []

sc = SparkContext.getOrCreate()

# make new ta feng dataset
rddFromFile = sc.textFile(str(sys.argv[3])).map(lambda line: line.split(","))
header = rddFromFile.first()
rddFromFile = rddFromFile.filter(lambda row: row != header) # filter out the header

rdd = rddFromFile.map(lambda x : ('-'.join([x[0].replace('"',''), x[1].replace('"','')]), int(x[5].replace('"','')))).collect()

with open('processed.txt', 'w+') as fout:
    writer =  csv.writer(fout)
    writer.writerow(['DATE-CUSTOMER_ID', 'PRODUCT_ID'])
    for line in rdd:
        writer.writerow(line)

# start timer
start_time = time.time()

# read in processed.txt
rddFromFile = sc.textFile('processed.txt')
header = rddFromFile.first()
rddFromFile = rddFromFile.filter(lambda row: row != header) # filter out the header

# filter out customers who bought more than k items
rddSplit = rddFromFile.map(lambda f: (f.split(',')[0], f.split(',')[1]))
rdd = rddSplit.groupByKey().mapValues(set).filter(lambda g: len(g[1]) > filter)

# get count of whole dataset
rdd_size = rdd.count()

# pass one function
def pass_one(p, rdd_size):
    frequents = [] # list of all frequent items

    chunk = [list(e[1]) for e in p]

    ps = math.ceil((len(chunk)/rdd_size)*support) # new threshold
    
    merged = list(itertools.chain.from_iterable(chunk)) # partition merged
    
    # calculate singles
    singles = dict()
    for x in merged:
        if(x not in singles):
            singles[x] = merged.count(x)
            
    frequent_items = []
    
    # get frequent singles
    for key,val in singles.items():
        if(val >= ps):
            frequent_items.append(key)
    
    frequent_items = sorted(frequent_items)
    single_tups = [(e,) for e in frequent_items]
    frequents.append(single_tups)
    
    frequent = True
    comb_val = 2
    fset = set(frequent_items)
    
    # start apriori
    while(frequent):
        # for each basket, find combinations with intersection of frequent items and basket
        data = dict()
        for basket in chunk:
            li = sorted(set(basket) & set(fset))
            basket_combs = list(combinations(li, comb_val))
            for bc in basket_combs:
                bc = tuple(bc)
                if(bc not in data):
                    data[bc] = 1
                else:
                    data[bc] = data[bc] + 1
                        
        frequent_items = []

        # check support threshold
        for key,val in data.items():
            if(val >= ps):
                frequent_items.append(key)
        
        frequent_items = sorted(frequent_items)
        frequents.append(frequent_items)
        
        comb_val = comb_val + 1 # increase combination value
        
        if(len(frequent_items) == 0):
            frequent = False
        
        # create new combination set
        fset = set()
        for f in frequent_items:
            f = set(f)
            fset = fset | f
    
    print(frequents)
    return frequents
    
# pass 1
candidates = list(set(itertools.chain.from_iterable(rdd.mapPartitions(lambda p: pass_one(p, rdd_size)).collect())))
candidates.sort(key = len)

# pass 2
baskets = rdd.map(lambda v: v[1]).collect()
itemsets = dict()
for basket in baskets:
    for candidate in candidates:
        if(set(candidate).issubset(basket)):
            if(candidate in itemsets):
                itemsets[candidate] = itemsets[candidate] + 1
            else:
                itemsets[candidate] = 1

frequent_items = sc.parallelize(list(itemsets.items())).filter(lambda t: t[1] >= support).map(lambda f: f[0]).collect()
frequent_items.sort(key = len)

# write to file
with open(str(sys.argv[4]), 'w') as f:
    f.write("Candidates:\n")
    
    max_value = len(candidates[-1])
    candidates_list = []
    for x in range(1, max_value+1):
        temp_list = [e for e in candidates if len(e) == x]
        temp_list = sorted(temp_list)
        candidates_list.append(temp_list)
        
    for candidate in candidates_list:
        f.write(str(candidate).replace("[", "").replace("]","").replace(",)",")").replace("), (", "),("))
        f.write("\n\n")
    
    max_value = len(frequent_items[-1])
    f_list = []
    for x in range(1,max_value+1):
        temp_list = [e for e in frequent_items if len(e) == x]
        temp_list = sorted(temp_list)
        f_list.append(temp_list)
        
    f.write("Frequent Itemsets:\n")
    for frequent in f_list:
        f.write(str(frequent).replace("[", "").replace("]","").replace(",)",")").replace("), (", "),("))
        f.write("\n\n")
        
duration = time.time() - start_time

print("Duration: ", duration)

