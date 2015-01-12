#encoding:utf-8
import KNN

f = open('data.txt')

data = []
tags = []
for line in f.readlines():
	line = line.split()
	tmp = line[:3]
	tmp = [ int(item) for item in tmp ]
	data.append(tmp)
	tags.append(line[-1])
f.close()

print(len(data))
trainData = data[:7000]
trainTags = tags[:7000]
testingData = data[7001:]
testingTags = tags[7001:]

knn = KNN.KNN(3,trainData,trainTags)

hit , error = 0 , 0 
for i,test in enumerate(testingData) :
	if testingTags[i] == knn.main(test) :
		hit += 1
		print(True)
	else :
		error += 1
		print(False)
	knn.clearResult()

print( float(hit) / (error+hit) )
	
