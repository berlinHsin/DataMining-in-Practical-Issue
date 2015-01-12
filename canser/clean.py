#encoding:utf-8

import decimal

info = open('test_info.txt')
feature = open('test_Features.txt')

infoData , featureData , result = [] , [] , []

for line in info.readlines() :
	line = line.split()
	tag = 1
	if float(line[0]) == -1 :
		tag = 0
	data = line[-4:] + [str(tag)]
	infoData.append(data)
info.close()

for line in feature.readlines() :
	line = line.split()
	featureData.append(line)
feature.close()

for i , data in enumerate(infoData) :
	tmp = featureData[i] + data
	result.append(tmp)

r = open('testing.txt','w')

r.write("@RELATION canser\n")
for i in range(len(result[0])-1) :
	r.write("@ATTRIBUTE f{} NUMERIC\n".format(i))
r.write("@ATTRIBUTE tag {0,1}\n")
r.write("@DATA\n")

for line in result :
	output = " ".join(line)
	output+= "\n"
	r.write(output)
r.close()
