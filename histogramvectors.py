from collections import OrderedDict, defaultdict
from sklearn.svm import LinearSVC #Import LinearSVC or SVC depending on the type of classifier
from sklearn.metrics import f1_score
import numpy as np

biz_id_list_train = ['344', '345', '346', '347', '340', '3999', '3998', '3994', '348', '3992', '2919', '2319', '2317', '2315', '2312', '298', '299', '297', '292', '293', '290', '291', '3772', '3777', '3775', '3774', '3779', '270', '271', '272', '275', '276', '278', '2263', '2260', '2266', '2267', '2265', '2443', '2444', '2445', '108', '109', '103', '100', '101', '104', '105', '2046', '2047', '2043', '2048', '2049', '3150', '2038', '3519', '3511', '2688', '2689', '2685', '2686', '2687', '2680', '2681', '2683', '342', '645', '99', '91', '93', '96', '1991', '3206', '3205', '1620', '1627', '1626', '3201', '1624', '3208', '2861', '2864', '558', '2869', '556', '1502', '1198', '3348', '3342', '1190', '1193', '3341', '3347', '1759', '1758', '1754', '1753', '1751', '1750', '1177', '1176', '1175', '1172', '1171', '1179', '510', '1284', '1281', '1283', '1282', '1289', '876', '1579', '689', '688', '685', '1572', '1570', '1577', '680', '683', '682', '459', '621', '873', '626', '1243', '407', '1370', '405', '1375', '402', '401', '400', '1379', '1378', '1342', '408', '3835', '3836', '3831', '3833', '3839', '3207', '456', '371', '370', '373', '372', '374', '377', '376', '1244', '393', '391', '390', '397', '396', '395', '394', '399', '398', '2308', '2301', '2300', '2303', '2302', '2305', '2306', '3748', '3746', '3747', '3743', '3741', '245', '244', '247', '241', '240', '242', '248', '3202', '2454', '2457', '2456', '2451', '2450', '179', '177', '175', '172', '171', '3699', '2052', '2056', '3690', '2058', '3692', '3693', '3696']
biz_id_list_test = ['3697', '3596', '1505', '3525', '3522', '3523', '3521', '3528', '3529', '3083', '2699', '2697', '2696', '2693', '2691', '2690', '1602', '1605', '3549', '3222', '3232', '3230', '1968', '3237', '1618', '3235', '1617', '1966', '1615', '1613', '1962', '1611', '2872', '2878', '3354', '3350', '3353', '3352', '1768', '1762', '1763', '1760', '1767', '1765', '1142', '1140', '1145', '1149', '692', '1544', '1545', '694', '695', '698', '543', '540', '541', '546', '547', '549', '761', '2841', '415', '416', '417', '1388', '1389', '412', '413', '1386', '1387', '1382', '3827', '3826', '3824', '3823', '3822', '3820', '3582', '3828', '368', '3589', '364', '362', '363', '361', '385', '387', '388', '1077', '3751', '3750', '3753', '3752', '3754', '2191', '2192', '2193', '2197', '3217', '251']

biz_to_count = OrderedDict()
train_vectors = defaultdict(list)
hist_vectors_train = []
hist_vectors_test = []
clust_lab_train = []
clust_lab_train_photo_id = []
clust_lab_test = []
clust_lab_test_photo_id = []

d = defaultdict(list)
d2 = {}
d3 = defaultdict(list)

biz_id_list_train_2 = []
biz_id_list_test_2 = []

train_vector_a0 = []
train_vector_a1 = []
train_vector_a2 = []
train_vector_a3 = []
train_vector_a4 = []
train_vector_a5 = []
train_vector_a6 = []
train_vector_a7 = []
train_vector_a8 = []

test_vector_a0 = []
test_vector_a1 = []
test_vector_a2 = []
test_vector_a3 = []
test_vector_a4 = []
test_vector_a5 = []
test_vector_a6 = []
test_vector_a7 = []
test_vector_a8 = []

with open('train.csv','r') as f: #the training data provided by kaggle - lists each bag(biz_id) and its associated attributes 
    count = 0
    for l in f:
        if count==0:
            count = 1
            continue
        else:
            l = l.strip()
            line_each = l.split(',')
            biz_to_count[count]=line_each[0]
            indi_list = line_each[1].split(" ")
            #print indi_list
            for j in range(0,9):
                if str(j) in indi_list:
                    train_vectors[j].append(1)
                else:
                    train_vectors[j].append(0)
            count = count + 1
f.close()

#print train_vectors    

for key in biz_to_count.keys():
    if biz_to_count[key] in biz_id_list_train:
        biz_id_list_train_2.append(biz_to_count[key])

#print biz_id_list_train_2

for key in biz_to_count.keys():
    if biz_to_count[key] in biz_id_list_test:
        biz_id_list_test_2.append(biz_to_count[key])

#print biz_id_list_test_2


with open('train_photo_to_biz_ids.csv','r') as f: #Dataset from kaggle - list of all images inside a bag/biz_id
    for l in f:
        l = l.rstrip()
        line_each = l.split(',')
        d[line_each[1]].append(line_each[0])
f.close()

with open('train_clust_labels.txt','r') as f: #Output of R file - cluster labels for training set
    for l in f:
        l = l.strip()
        line = l.split(',')
        clust_lab_train.append(line[0])
        clust_lab_train_photo_id.append(line[1])
f.close()

for key in biz_id_list_train_2:
    list_temp = [0]*196
    value = d[key]    
    for each in value:
        if each in  clust_lab_train_photo_id:
            ind = clust_lab_train_photo_id.index(each)
            list_temp[int(clust_lab_train[ind]) - 1] = list_temp[int(clust_lab_train[ind]) - 1] + 1
    hist_vectors_train.append(list_temp)
#    print 'one key done'

num = 0
with open('test_clust_labels','r') as f: #Output of r file - cluster labels for validation set
    for l in f:
        l = l.strip()
        line = l.split(',')
        clust_lab_test.append(line[0])
        partition = line[1].split('.')
        clust_lab_test_photo_id.append(partition[0])
        num = num + 1
f.close()

for key in biz_id_list_test_2:
    list_temp = [0]*196
    value = d[key]    
    for each in value:
        if each in  clust_lab_test_photo_id:
            ind = clust_lab_test_photo_id.index(each)
            list_temp[int(clust_lab_test[ind]) - 1] = list_temp[int(clust_lab_test[ind]) - 1] + 1
    hist_vectors_test.append(list_temp)

train_vector_0 = train_vectors[0]    
train_vector_1 = train_vectors[1]    
train_vector_2 = train_vectors[2]    
train_vector_3 = train_vectors[3]    
train_vector_4 = train_vectors[4]    
train_vector_5 = train_vectors[5]    
train_vector_6 = train_vectors[6]    
train_vector_7 = train_vectors[7]    
train_vector_8 = train_vectors[8]

for biz_id in biz_id_list_train_2:
    ind = biz_to_count.keys()[biz_to_count.values().index(biz_id)]
    ind = ind - 1
    train_vector_a0.append(train_vector_0[ind])
    train_vector_a1.append(train_vector_1[ind])
    train_vector_a2.append(train_vector_2[ind])
    train_vector_a3.append(train_vector_3[ind])
    train_vector_a4.append(train_vector_4[ind])
    train_vector_a5.append(train_vector_5[ind])
    train_vector_a6.append(train_vector_6[ind])
    train_vector_a7.append(train_vector_7[ind])
    train_vector_a8.append(train_vector_8[ind])
    
X = np.array(hist_vectors_train)
y0 = np.array(train_vector_a0)
y1 = np.array(train_vector_a1)
y2 = np.array(train_vector_a2)
y3 = np.array(train_vector_a3)
y4 = np.array(train_vector_a4)
y5 = np.array(train_vector_a5)
y6 = np.array(train_vector_a6)
y7 = np.array(train_vector_a7)
y8 = np.array(train_vector_a8)
Z = np.array(hist_vectors_test)
#If using SVC(), say clf0 = SVC() and so on 
clf0 = LinearSVC()
clf1 = LinearSVC()
clf2 = LinearSVC()
clf3 = LinearSVC()
clf4 = LinearSVC()
clf5 = LinearSVC()
clf6 = LinearSVC()
clf7 = LinearSVC()
clf8 = LinearSVC()

clf0.fit(X, y0)
clf1.fit(X, y1)
clf2.fit(X, y2)
clf3.fit(X, y3)
clf4.fit(X, y4)
clf5.fit(X, y5)
clf6.fit(X, y6)
clf7.fit(X, y7)
clf8.fit(X, y8)

res_0 = (clf0.predict(Z)).tolist()
res_1 = (clf1.predict(Z)).tolist()
res_2 = (clf2.predict(Z)).tolist()
res_3 = (clf3.predict(Z)).tolist()
res_4 = (clf4.predict(Z)).tolist()
res_5 = (clf5.predict(Z)).tolist()
res_6 = (clf6.predict(Z)).tolist()
res_7 = (clf7.predict(Z)).tolist()
res_8 = (clf8.predict(Z)).tolist()

for biz_id in biz_id_list_test_2:
    ind = biz_to_count.keys()[biz_to_count.values().index(biz_id)]
    ind = ind - 1
    test_vector_a0.append(train_vector_0[ind])
    test_vector_a1.append(train_vector_1[ind])
    test_vector_a2.append(train_vector_2[ind])
    test_vector_a3.append(train_vector_3[ind])
    test_vector_a4.append(train_vector_4[ind])
    test_vector_a5.append(train_vector_5[ind])
    test_vector_a6.append(train_vector_6[ind])
    test_vector_a7.append(train_vector_7[ind])
    test_vector_a8.append(train_vector_8[ind])

c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 0
c7 = 0
c8 = 0

for i in range(0,101):
    if res_0[i] == test_vector_a0[i]:
        c0 = c0 + 1
    if res_1[i] == test_vector_a1[i]:
        c1 = c1 + 1
    if res_2[i] == test_vector_a2[i]:
        c2 = c2 + 1
    if res_3[i] == test_vector_a3[i]:
        c3 = c3 + 1
    if res_4[i] == test_vector_a4[i]:
        c4 = c4 + 1
    if res_5[i] == test_vector_a5[i]:
        c5 = c5 + 1
    if res_6[i] == test_vector_a6[i]:
        c6 = c6 + 1
    if res_7[i] == test_vector_a7[i]:
        c7 = c7 + 1
    if res_8[i] == test_vector_a8[i]:
        c8 = c8 + 1

print "Accuracy of classificaion for attribute 0 is ", c0
print "Accuracy of classificaion for attribute 1 is ", c1
print "Accuracy of classificaion for attribute 2 is ", c2
print "Accuracy of classificaion for attribute 3 is ", c3
print "Accuracy of classificaion for attribute 4 is ", c4
print "Accuracy of classificaion for attribute 5 is ", c5
print "Accuracy of classificaion for attribute 6 is ", c6
print "Accuracy of classificaion for attribute 7 is ", c7
print "Accuracy of classificaion for attribute 8 is ", c8
print "\n"
print "F-measure of classificaion for attribute 0 is ", f1_score(test_vector_a0, res_0)
print "F-measure of classificaion for attribute 1 is ", f1_score(test_vector_a1, res_1)
print "F-measure of classificaion for attribute 2 is ", f1_score(test_vector_a2, res_2)
print "F-measure of classificaion for attribute 3 is ", f1_score(test_vector_a3, res_3)
print "F-measure of classificaion for attribute 4 is ", f1_score(test_vector_a4, res_4)
print "F-measure of classificaion for attribute 5 is ", f1_score(test_vector_a5, res_5)
print "F-measure of classificaion for attribute 6 is ", f1_score(test_vector_a6, res_6)
print "F-measure of classificaion for attribute 7 is ", f1_score(test_vector_a7, res_7)
print "F-measure of classificaion for attribute 8 is ", f1_score(test_vector_a8, res_8)


    
            
        
        
