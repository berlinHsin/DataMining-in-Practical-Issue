#encoding:utf-8
def testing(value) :
	try :
		return value == int(value)
	except :
		return False

f = open('customer.txt')
o = open('out.txt','w')

for line in f.readlines() :
	data = line.split()

	income = "2"
	if int(data[-1]) < 60000 :
		income = "1"
	if int(data[-1]) >= 90000 :
		income = "3"
	
	age = "2"
	if int(data[-2]) < 40 :
		age = "1"
	if int(data[-2]) > 60 :
		age = "3"

	if data[-8] == "M" or data[-8] == "F" :
		if data[-8] == "M" :
			gender = 0
		else :
			gender = 1
		text = "{} {} {} {} \n".format(income,age,gender,data[-3])
	else :
		if data[-9] == "M" :
			gender = 0 
		else :
			gender = 1
		text = "{} {} {} {} \n".format(income,age,gender,data[-3])
	o.write(text)

f.close()
o.close()
