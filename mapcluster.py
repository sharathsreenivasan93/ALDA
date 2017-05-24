# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 01:35:02 2016

@author: sharathsreenivasan
"""
from scipy import spatial

line_all_cluster = []
# Read the file which contains the cluster labels
with open('centersnlabels.txt','r') as f:
    for l in f:
        line = [float(x) for x in l.split(',')]
        line_all_cluster.append(line[:-1])

line_all_feature = []
#Read the file which contains the output of the feature extraction python code
with open('output_extra.txt','r') as f:
    for l in f:
        line = [float(x) for x in l.split()]
        line_all_feature.append(line)

for i in line_all_feature:
    if len(i)<1024:
        print "WRONG", line_all_feature.index(i), len(i)        
print len(line_all_cluster[1])

print len(line_all_feature[0])

#result = 1 - spatial.distance.cosine(line_all_cluster[3],line_all_cluster[0])
count = 0
for i in line_all_feature[:-1]:
    minimum = 999
    pos = 0
    for j in line_all_cluster:
        distance = 1 - spatial.distance.cosine(i[1:],j)
        if distance < minimum:
            minimum = distance
            pos = line_all_cluster.index(j)
    count = count + 1
    print i[0], pos, count


#print result
