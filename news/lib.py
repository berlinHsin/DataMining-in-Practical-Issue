#encoding:utf-8

from stemming.porter2 import stem
import collections ,time , random 
import numpy as np 

class PreProcessing :
	def __init__( self ) :
		self.Vocab = []
		self.Docs  = []
		self.Mm    = []
		self.tags  = []
	

	def addVocab( self , doc , tag) :
		""" ( string , string) -> None """
		text = []
		content = [ stem(word) for word in doc.split() ]
		length  = len(content)
		content = collections.Counter(content).most_common()
		for item in content :
			if item[0] not in self.Vocab :
				self.Vocab.append(item[0])
			index = self.Vocab.index(item[0])
			count = float(item[1])/length
			text.append( (index,count) )
		self.Docs.append(text)
		self.tags.append(tag)
	
	def setMm( self ) :
		""" None -> None """
		self.Mm = np.zeros( ( len(self.Docs) , len(self.Vocab) ) )
		for i , doc in enumerate(self.Docs) :
			for item in doc  :
				index , count = item[0] , item[1]
				self.Mm[i][index] = count
	
	def getVocab( self ) :
		return self.Vocab 

	def getDocs( self ):
		return self.Docs 
	
	def getMm( self ):
		return self.Mm

	def getTags(self):
		return self.tags

class KNN :

	def __init__( self , K , Datas , tags,samplingCount,features = 100) :
		self.K = K
		self.Datas = Datas
		self.tags  = tags
		self.samplingCount = samplingCount
		self.features = features
		self.Order = []
	
	def getLength(self,lst) :
		""" (list) -> float """
		length = 0.
		for i in lst :
			length += i**(2)
		return length**(0.5) if length> 0 else 1
	
	def splitCount(self,sentence) :
		#sentence = sentence.split()
		return collections.Counter(sentence).most_common()[:self.features]

	def tokenize(self,Vocab,vector) :
		result = []
		length = len(vector)
		for item in vector :
			if item[0] not in Vocab :
				Vocab.append(item[0])
			index = Vocab.index(item[0])
			count = float(item[1])/length
			result.append( (index,count))
		return (Vocab,result)

	def formMm(self,sentenceA,sentenceB) :
		Vocab = []
		vectorA , vectorB = self.splitCount(sentenceA),self.splitCount(sentenceB)
		Vocab , vectorA = self.tokenize(Vocab,vectorA)
		Vocab , vectorB = self.tokenize(Vocab,vectorB)
		docs = [vectorA,vectorB]
		Mm = np.zeros((2,len(Vocab)))
		for i , doc in enumerate(docs) :
			for item in doc :
				index , count = item[0] , item[1]
				Mm[i][index] = count
		return Mm

	def euDistance(self,lstA,lstB) :
		summation = 0.
		for i,value in enumerate(lstA) :
			summation += (value-lstB[i])**2
		return summation**(0.5)


	def calDistance(self,lstA,lstB) :
		""" (lst ,lst) -> float """
		summation = 0.
		for i,value in enumerate(lstA) :
			summation += value*lstB[i]
		return summation/(self.getLength(lstA)*self.getLength(lstB))

	def check( self, simScore , tag ) :
		self.Order.append( (tag,simScore))

	def classification(self,lst) :
		#s = time.time()
		sampling = random.sample(range(len(self.Datas)),self.samplingCount)
		### cosine sim ###
		"""
		for i in sampling :
			Mm = self.formMm(lst,self.Datas[i])
			distance = self.calDistance(Mm[0],Mm[1])
			self.check(distance,self.tags[i])
		"""
		### eudistance ###
		for i in sampling :
			Mm = self.formMm(lst,self.Datas[i])
			distance = self.euDistance(Mm[0],Mm[1])
			self.check(distance,self.tags[i])
		

		result = self.getResult() 
		self.clearResult()
		#print time.time()-s
		return result 
	
	def clearResult(self) :
		self.Order = []
	
	def getResult( self ) :
		### cosine sim ###
		self.Order = sorted(self.Order,key=lambda x : x[1],reverse=True)[0:self.K]
		result = self.Order[0][0]
		return result 
		### euDistance###	
		"""
		self.Order = sorted(self.Order,key=lambda x : x[1])[0:self.K]
		result = self.Order[0][0]
		return result 
		"""

class KMEANS :
	def __init__(self,K,datas,iters=50) :
		self.K = K
		self.Datas = datas 
		self.Seed = []
		self.Result = []
		self.iters  = iters

	def getLength(self,lst) :
		""" (list) -> float """
		length = 0.
		for i in lst :
			length += i**(2)
		return length**(0.5) if length> 0 else 1

	def randomPick(self) :
		randLst = random.sample(range(len(self.Datas)),self.K)
		self.Seed = [ self.Datas[index] for index in randLst]

	def calDistance(self,lstA,lstB) :
		""" (lst ,lst) -> float """
		summation = 0.
		for i,value in enumerate(lstA) :
			summation += value*lstB[i]
			#summation += (value - lstB[i] )**2
		return summation/(self.getLength(lstA)*self.getLength(lstB))

	def euDistance(self,lstA,lstB) :
		summation = 0.
		for i,value in enumerate(lstA) :
			summation += (value-lstB[i])**2
		return summation**(0.5)

	def nearestCore(self,data) :
		core = 0 
		nearest = 100
		#nearest = 0 
		### cosine sim ###
		"""
		for i,seed in enumerate(self.Seed) :
			distance = self.calDistance(seed,data)
			if distance > nearest  :
				nearest = distance 
				core = i
		"""
		for i , seed in enumerate(self.Seed) :
			distance = self.euDistance(seed,data)
			if distance < nearest :
				nearest = distance
				core = i
		return core

	def calCore(self,group) :
		summation = []
		for i in range(len(self.Datas[0])) :
			summation.append(0)
		for index in group :
			summation = [ (x + y) for x , y in zip(summation,self.Datas[index])]
		count = len(group)
		for i ,value in enumerate(summation):
			summation[i] = float(value)/count
		return summation

	def seprate(self) :
		self.initResult()
		for i , data in enumerate(self.Datas) :
			core = self.nearestCore(data)
			self.Result[core].append(i)

	def initResult(self) :
		for i in range(self.K) :
			if i >= len(self.Result) :
				self.Result.append([])
			else :
				self.Result[i] = []

	def classifier(self) : 
		self.initResult()
		self.randomPick()
		self.seprate()
		for count in range(self.iters):
			print count
			for i , group in enumerate(self.Result) :
				newCore = self.calCore(group) 	
				self.Seed[i] = newCore
			self.seprate()
		#print(self.Result)
		cls = []
		for i in range(self.K) :
			cls.append(0)
		for group in self.Result :
			for element in group :
				if element < 50 :
					cls[0] += 1
				if element < 100 and element >=50 :
					cls[1] += 1
				if element < 150 and element >=100 :
					cls[2] += 1
				if element < 200 and element >=150 :
					cls[3] += 1
			print("cls1 : {} , cls2 : {} , cls3 :{} , cls4 :{}".format(cls[0],cls[1],cls[2],cls[3]))
			for i in range(self.K) :
				cls[i] = 0

class HIERARCHICAL :

	def __init__(self,K,Datas):
		self.K = K
		self.Datas = Datas 
		self.Result = []
		self.Group  = []
		self.trace  = []

	def calDistance(self,lstA,lstB) :
		""" (lst ,lst) -> float """
		summation = 0.
		for i,value in enumerate(lstA) :
			summation += value*lstB[i]
			#summation += (value - lstB[i] )**2
		return summation/(self.getLength(lstA)*self.getLength(lstB))

	def getLength(self,lst) :
		""" (list) -> float """
		length = 0.
		for i in lst :
			length += i**(2)
		return length**(0.5) if length> 0 else 1
	
	def getRestNearest(self,index,lst) :
		### cosine sim ###

		nearest = 0 
		maximum = 0.
		for i in lst :
			if i != index :
				distance = self.calDistance(self.Group[index],self.Group[i])
				if distance > maximum :
					nearest , maximum = i , distance 
		return nearest
		"""
		nearest = 0
		minimum = 1000
		for i in lst :
			if i != index :
				distance = self.euDistance(self.Group[index],self.Group[i])
				if distance < minimum :
					nearest , minimum = i , distance 
		return nearest
		"""

	def calCore(self,lst) :
		""" ( [1,42,321] ) -> ( [0.52,0.1423] ) """
		summation = []
		for i , index in enumerate(lst) :
			for j , element in enumerate(self.Datas[index]) :
				if i == 0 :
					summation.append(element)
				else :
					summation[j] += element
		length = len(lst)
		for i , element in enumerate(summation) :
			summation[i] = float(element)/length

		return summation

	def euDistance(self,lstA,lstB) :
		summation = 0.
		for i,value in enumerate(lstA) :
			summation += (value-lstB[i])**2
		return summation**(0.5)

	def merge(self) :
		randLst = random.sample(range(len(self.Group)),len(self.Group))
		removeLst = []
		for groupId in randLst :
			nearest = self.getRestNearest(groupId,randLst)
			randLst.remove(nearest)
			self.Result[groupId]+=self.Result[nearest]
			removeLst.append(self.Result[nearest])

		for element in removeLst :
			self.Result.remove(element)
		
		print(self.Result)
		newCore = []
		for group in self.Result :
			newCore.append(self.calCore(group))
		self.Group = newCore


	def classifier(self):
		self.Group = self.Datas 
		for i in range(len(self.Group)) :
			self.Result.append([i])
		while len(self.Group)>self.K :
			self.merge()
		#print(self.Result)
		cls = []
		for i in range(self.K) :
			cls.append(0)
		for group in self.Result :
			for element in group :
				if element < 50 :
					cls[0] += 1
				if element < 100 and element >=50 :
					cls[1] += 1
				if element < 150 and element >=100 :
					cls[2] += 1
				if element < 200 and element >=150 :
					cls[3] += 1
			print("cls1 : {} , cls2 : {} , cls3 {} , cls4 {}".format(cls[0],cls[1],cls[2],cls[3]))
			for i in range(self.K) :
				cls[i] = 0

if __name__ == "__main__" :
	pass
