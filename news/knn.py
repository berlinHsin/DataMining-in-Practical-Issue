#encoding : utf-8

from library import lib

class KNN(lib) :

	def __init__(self, K , Datas , tags) :
		self.K = K
		self.Datas = Datas 
		self.tags  = tags
		self.Result = []
		self.distanceType = self.cosineSim
		self.sortReverse  = True
	
	def getResult(self) :
		self.Result = sorted(self.Result,key=lambda x : x[0],reverse=self.sortReverse)[0:self.K]
		countLst = {}
		for data in self.Result :
			if data[1] in countLst :
				countLst[data[1]] += 1 
			else :
				countLst.update({data[1]:1})
		return max(countLst,key=countLst.get)

	def clearResult(self) :
		self.Result = []
	
	def main(self,lst) :
		""" 
			(list) -> string  
			We should first set datas into same dimension !
		"""
		for i,data in enumerate(self.Datas) :
			distance , check = self.distanceType(data,lst)  
			self.Result.append((distance,self.tags[i]))
		self.sortReverse = check(10,0)
		# see cosineSim and euDistance in library .
		# if check(10,0) is True : cosineSim , else : euDistance
		# And set sortReverse = True for cosineSim , because the higher score the closer.

		return self.getResult()

if __name__ == "__main__" :
	exampleLst = [ [0,24,5] , [24,1,0] , [0,1,0] , [0,25,2] , [25,0,2]]
	tags       = ['a','b','a','a','b']
	obj = KNN(2,exampleLst,tags)
	print(obj.main([4,20,1]))
	obj.clearResult()
	print(obj.main([24,2,1]))

