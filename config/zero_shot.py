
################### 75:25 Split #######################

split_t1_train_0 = [] 
split_t1_test_0 = [] 
split_t1_train_1 = []
split_t1_test_1 = []
split_t1_train_2 = []
split_t1_test_2 = []
split_t1_train_3 = []
split_t1_test_3 = []
split_t1_train_4 = []
split_t1_test_4 = []
split_t1_train_5 = []
split_t1_test_5 = []
split_t1_train_6 = []
split_t1_test_6 = []
split_t1_train_7 = []
split_t1_test_7 = []
split_t1_train_8 = []
split_t1_test_8 = []
split_t1_train_9 = []
split_t1_test_9 = []

with open('splits/train_75_test_25/ActivityNet/train/split_0.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_0.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_0.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_0.append(files1[:-1])


with open('splits/train_75_test_25/ActivityNet/train/split_1.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_1.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_1.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_1.append(files1[:-1])
    
    
with open('splits/train_75_test_25/ActivityNet/train/split_2.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_2.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_2.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_2.append(files1[:-1])
    
    
with open('splits/train_75_test_25/ActivityNet/train/split_3.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_3.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_3.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_3.append(files1[:-1])
    
with open('splits/train_75_test_25/ActivityNet/train/split_4.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_4.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_4.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_4.append(files1[:-1])
    
with open('splits/train_75_test_25/ActivityNet/train/split_5.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_5.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_5.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_5.append(files1[:-1])
    
    
with open('splits/train_75_test_25/ActivityNet/train/split_6.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_6.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_6.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_6.append(files1[:-1])
    
    
    
with open('splits/train_75_test_25/ActivityNet/train/split_7.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_7.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_7.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_7.append(files1[:-1])
    
with open('splits/train_75_test_25/ActivityNet/train/split_8.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_8.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_8.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_8.append(files1[:-1])
    
with open('splits/train_75_test_25/ActivityNet/train/split_9.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/ActivityNet/test/split_9.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()

for files in filecontents:
    split_t1_train_9.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_9.append(files1[:-1])

################### 50:50 Split #######################

split_t2_train_0 = [] 
split_t2_test_0 = [] 
split_t2_train_1 = []
split_t2_test_1 = []
split_t2_train_2 = []
split_t2_test_2 = []
split_t2_train_3 =[]
split_t2_test_3 = []
split_t2_train_4 = []
split_t2_test_4 = []
split_t2_train_5 = []
split_t2_test_5 = []
split_t2_train_6 = []
split_t2_test_6 = []
split_t2_train_7 = []
split_t2_test_7 = []
split_t2_train_8 = []
split_t2_test_8 = []
split_t2_train_9 = []
split_t2_test_9 = []




with open('splits/train_50_test_50/ActivityNet/train/split_0.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_0.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_0.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_0.append(files3[:-1])


t1_dict_train_0 = {split_t1_train_0[i] : i for i in sorted(range(150))}
t1_dict_test_0 = {split_t1_test_0[i] : i for i in sorted(range(50))}

t2_dict_train_0 = {split_t2_train_0[i] : i for i in sorted(range(100))}
t2_dict_test_0 = {split_t2_test_0[i] : i for i in sorted(range(100))}



with open('splits/train_50_test_50/ActivityNet/train/split_1.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_1.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_1.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_1.append(files3[:-1])



t1_dict_train_1 = {split_t1_train_1[i] : i for i in sorted(range(150))}
t1_dict_test_1 = {split_t1_test_1[i] : i for i in sorted(range(50))}

t2_dict_train_1 = {split_t2_train_1[i] : i for i in sorted(range(100))}
t2_dict_test_1 = {split_t2_test_1[i] : i for i in sorted(range(100))}


with open('splits/train_50_test_50/ActivityNet/train/split_2.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_2.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_2.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_2.append(files3[:-1])



t1_dict_train_2 = {split_t1_train_2[i] : i for i in sorted(range(150))}
t1_dict_test_2 = {split_t1_test_2[i] : i for i in sorted(range(50))}

t2_dict_train_2 = {split_t2_train_2[i] : i for i in sorted(range(100))}
t2_dict_test_2 = {split_t2_test_2[i] : i for i in sorted(range(100))}



with open('splits/train_50_test_50/ActivityNet/train/split_3.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_3.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_3.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_3.append(files3[:-1])



t1_dict_train_3 = {split_t1_train_3[i] : i for i in sorted(range(150))}
t1_dict_test_3 = {split_t1_test_3[i] : i for i in sorted(range(50))}

t2_dict_train_3 = {split_t2_train_3[i] : i for i in sorted(range(100))}
t2_dict_test_3 = {split_t2_test_3[i] : i for i in sorted(range(100))}


with open('splits/train_50_test_50/ActivityNet/train/split_4.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_4.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_4.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_4.append(files3[:-1])



t1_dict_train_4 = {split_t1_train_4[i] : i for i in sorted(range(150))}
t1_dict_test_4 = {split_t1_test_4[i] : i for i in sorted(range(50))}

t2_dict_train_4 = {split_t2_train_4[i] : i for i in sorted(range(100))}
t2_dict_test_4 = {split_t2_test_4[i] : i for i in sorted(range(100))}


with open('splits/train_50_test_50/ActivityNet/train/split_5.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_5.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_5.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_5.append(files3[:-1])



t1_dict_train_5 = {split_t1_train_5[i] : i for i in sorted(range(150))}
t1_dict_test_5 = {split_t1_test_5[i] : i for i in sorted(range(50))}

t2_dict_train_5 = {split_t2_train_5[i] : i for i in sorted(range(100))}
t2_dict_test_5 = {split_t2_test_5[i] : i for i in sorted(range(100))}

with open('splits/train_50_test_50/ActivityNet/train/split_6.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_6.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_6.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_6.append(files3[:-1])



t1_dict_train_6 = {split_t1_train_6[i] : i for i in sorted(range(150))}
t1_dict_test_6 = {split_t1_test_6[i] : i for i in sorted(range(50))}

t2_dict_train_6 = {split_t2_train_6[i] : i for i in sorted(range(100))}
t2_dict_test_6 = {split_t2_test_6[i] : i for i in sorted(range(100))}


with open('splits/train_50_test_50/ActivityNet/train/split_7.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_7.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_7.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_7.append(files3[:-1])



t1_dict_train_7 = {split_t1_train_7[i] : i for i in sorted(range(150))}
t1_dict_test_7 = {split_t1_test_7[i] : i for i in sorted(range(50))}

t2_dict_train_7 = {split_t2_train_7[i] : i for i in sorted(range(100))}
t2_dict_test_7 = {split_t2_test_7[i] : i for i in sorted(range(100))}



with open('splits/train_50_test_50/ActivityNet/train/split_8.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_8.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_8.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_8.append(files3[:-1])



t1_dict_train_8 = {split_t1_train_8[i] : i for i in sorted(range(150))}
t1_dict_test_8 = {split_t1_test_8[i] : i for i in sorted(range(50))}

t2_dict_train_8 = {split_t2_train_8[i] : i for i in sorted(range(100))}
t2_dict_test_8 = {split_t2_test_8[i] : i for i in sorted(range(100))}

with open('splits/train_50_test_50/ActivityNet/train/split_9.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/ActivityNet/test/split_9.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()

for files2 in filecontents2:
    split_t2_train_9.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_9.append(files3[:-1])



t1_dict_train_9 = {split_t1_train_9[i] : i for i in sorted(range(150))}
t1_dict_test_9 = {split_t1_test_9[i] : i for i in sorted(range(50))}

t2_dict_train_9 = {split_t2_train_9[i] : i for i in sorted(range(100))}
t2_dict_test_9 = {split_t2_test_9[i] : i for i in sorted(range(100))}





################### 75:25 Split #######################

#  THUMOS14

split_t1_train_thumos_0 = [] 
split_t1_test_thumos_0 = [] 
split_t1_train_thumos_1 = [] 
split_t1_test_thumos_1 = [] 
split_t1_train_thumos_2 = [] 
split_t1_test_thumos_2 = [] 
split_t1_train_thumos_3 = [] 
split_t1_test_thumos_3 = [] 
split_t1_train_thumos_4 = [] 
split_t1_test_thumos_4 = [] 
split_t1_train_thumos_5 = [] 
split_t1_test_thumos_5 = [] 
split_t1_train_thumos_6 = [] 
split_t1_test_thumos_6 = []
split_t1_train_thumos_7 = [] 
split_t1_test_thumos_7 = []
split_t1_train_thumos_8 = [] 
split_t1_test_thumos_8 = []   
split_t1_train_thumos_9 = [] 
split_t1_test_thumos_9 = [] 




with open('splits/train_75_test_25/THUMOS14/train/split_0.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_0.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_0.append(files[:-1])
    
for files1 in filecontents1:
    split_t1_test_thumos_0.append(files1[:-1])
    
    
    
with open('splits/train_75_test_25/THUMOS14/train/split_1.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_1.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_1.append(files[:-1])
    
for files1 in filecontents1:
    split_t1_test_thumos_1.append(files1[:-1])



with open('splits/train_75_test_25/THUMOS14/train/split_2.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_2.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_2.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_2.append(files1[:-1])
    
    
    
with open('splits/train_75_test_25/THUMOS14/train/split_3.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_3.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_3.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_3.append(files1[:-1])
    
    
    
with open('splits/train_75_test_25/THUMOS14/train/split_4.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_4.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_4.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_4.append(files1[:-1])
    
    

with open('splits/train_75_test_25/THUMOS14/train/split_5.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_5.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_5.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_5.append(files1[:-1])



with open('splits/train_75_test_25/THUMOS14/train/split_6.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_6.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_6.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_6.append(files1[:-1])
    
    
with open('splits/train_75_test_25/THUMOS14/train/split_7.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_7.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_7.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_7.append(files1[:-1])
    

with open('splits/train_75_test_25/THUMOS14/train/split_8.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_8.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_8.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_8.append(files1[:-1])
    

with open('splits/train_75_test_25/THUMOS14/train/split_9.list', 'r') as filehandle:
    filecontents = filehandle.readlines()
with open('splits/train_75_test_25/THUMOS14/test/split_9.list', 'r') as filehandle:
    filecontents1 = filehandle.readlines()
    
for files in filecontents:
    split_t1_train_thumos_9.append(files[:-1])

for files1 in filecontents1:
    split_t1_test_thumos_9.append(files1[:-1])


################### 50:50 Split #######################


split_t2_train_thumos_0 = [] 
split_t2_test_thumos_0 = [] 
split_t2_train_thumos_1 = []
split_t2_test_thumos_1 = []
split_t2_train_thumos_2 = []
split_t2_test_thumos_2 = []
split_t2_train_thumos_3 = []
split_t2_test_thumos_3 = []
split_t2_train_thumos_4 = []
split_t2_test_thumos_4 = []
split_t2_train_thumos_5 = []
split_t2_test_thumos_5 = []
split_t2_train_thumos_6 = []
split_t2_test_thumos_6 = []
split_t2_train_thumos_7 = []
split_t2_test_thumos_7 = []
split_t2_train_thumos_8 = []
split_t2_test_thumos_8 = []
split_t2_train_thumos_9 = []
split_t2_test_thumos_9 = []
split_t2_train_thumos_10 = []
split_t2_test_thumos_10 = []








with open('splits/train_50_test_50/THUMOS14/train/split_0.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_0.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_0.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_0.append(files3[:-1])
    

t1_dict_train_thumos_0 = {split_t1_train_thumos_0[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_0 = {split_t1_test_thumos_0[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_0 = {split_t2_train_thumos_0[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_0 = {split_t2_test_thumos_0[i] : i for i in sorted(range(10))}



with open('splits/train_50_test_50/THUMOS14/train/split_1.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_1.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_1.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_1.append(files3[:-1])

t1_dict_train_thumos_1 = {split_t1_train_thumos_1[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_1 = {split_t1_test_thumos_1[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_1 = {split_t2_train_thumos_1[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_1 = {split_t2_test_thumos_1[i] : i for i in sorted(range(10))}


with open('splits/train_50_test_50/THUMOS14/train/split_2.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_2.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_2.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_2.append(files3[:-1])

t1_dict_train_thumos_2 = {split_t1_train_thumos_2[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_2 = {split_t1_test_thumos_2[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_2 = {split_t2_train_thumos_2[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_2 = {split_t2_test_thumos_2[i] : i for i in sorted(range(10))}


with open('splits/train_50_test_50/THUMOS14/train/split_3.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_3.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_3.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_3.append(files3[:-1])

t1_dict_train_thumos_3 = {split_t1_train_thumos_3[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_3 = {split_t1_test_thumos_3[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_3 = {split_t2_train_thumos_3[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_3 = {split_t2_test_thumos_3[i] : i for i in sorted(range(10))}


with open('splits/train_50_test_50/THUMOS14/train/split_4.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_4.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_4.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_4.append(files3[:-1])

t1_dict_train_thumos_4 = {split_t1_train_thumos_4[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_4 = {split_t1_test_thumos_4[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_4 = {split_t2_train_thumos_4[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_4 = {split_t2_test_thumos_4[i] : i for i in sorted(range(10))}



with open('splits/train_50_test_50/THUMOS14/train/split_5.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_5.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_5.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_5.append(files3[:-1])

t1_dict_train_thumos_5 = {split_t1_train_thumos_5[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_5 = {split_t1_test_thumos_5[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_5 = {split_t2_train_thumos_5[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_5 = {split_t2_test_thumos_5[i] : i for i in sorted(range(10))}



with open('splits/train_50_test_50/THUMOS14/train/split_6.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_6.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_6.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_6.append(files3[:-1])

t1_dict_train_thumos_6 = {split_t1_train_thumos_6[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_6 = {split_t1_test_thumos_6[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_6 = {split_t2_train_thumos_6[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_6 = {split_t2_test_thumos_6[i] : i for i in sorted(range(10))}



with open('splits/train_50_test_50/THUMOS14/train/split_7.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_7.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_7.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_7.append(files3[:-1])

t1_dict_train_thumos_7 = {split_t1_train_thumos_7[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_7 = {split_t1_test_thumos_7[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_7 = {split_t2_train_thumos_7[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_7 = {split_t2_test_thumos_7[i] : i for i in sorted(range(10))}



with open('splits/train_50_test_50/THUMOS14/train/split_8.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_8.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_8.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_8.append(files3[:-1])

t1_dict_train_thumos_8 = {split_t1_train_thumos_8[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_8 = {split_t1_test_thumos_8[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_8 = {split_t2_train_thumos_8[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_8 = {split_t2_test_thumos_8[i] : i for i in sorted(range(10))}



with open('splits/train_50_test_50/THUMOS14/train/split_9.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_9.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_9.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_9.append(files3[:-1])

t1_dict_train_thumos_9 = {split_t1_train_thumos_9[i] : i for i in sorted(range(15))}
t1_dict_test_thumos_9 = {split_t1_test_thumos_9[i] : i for i in sorted(range(5))}

t2_dict_train_thumos_9 = {split_t2_train_thumos_9[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_9 = {split_t2_test_thumos_9[i] : i for i in sorted(range(10))}


with open('splits/train_50_test_50/THUMOS14/train/split_10.list', 'r') as filehandle:
    filecontents2 = filehandle.readlines()
with open('splits/train_50_test_50/THUMOS14/test/split_10.list', 'r') as filehandle:
    filecontents3 = filehandle.readlines()
    
for files2 in filecontents2:
    split_t2_train_thumos_10.append(files2[:-1])

for files3 in filecontents3:
    split_t2_test_thumos_10.append(files3[:-1])

t2_dict_train_thumos_10 = {split_t2_train_thumos_10[i] : i for i in sorted(range(10))}
t2_dict_test_thumos_10 = {split_t2_test_thumos_10[i] : i for i in sorted(range(10))}