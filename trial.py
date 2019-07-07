
import csv 
import math

def entropy(setl):
	n=len(setl)
	unique=[]
	for x in setl:
		if x not in unique:
			unique.append(x)
	ent=0

	for x in unique:
		nx=0
		for y in setl:
			if x==y:
				nx=nx+1
		frac=float(nx)/float(n)
		ent-=frac*(math.log(frac,2))
	#print(ent)
	return ent
 
filename = "56.csv"
#answers=[]
		#print(filename)
fields = [] 
rows = []

with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader:
        rows.append([float(col) for col in row])  
numr=len(rows)
features=[]
numf=len(rows[0])
for x in range(0,numf):
	feature=[]
	for row in rows:
		feature.append(row[x])
	features.append(feature)
classv=[]
#classv.append(features[numf-1])
for x in features[numf-1]:
	classv.append(x)
ent=entropy(classv)
answer=[]

for feature in features:
	unique=[]
	for x in feature:
		if x not in unique:
			unique.append(x)
	
	#m=1
	values=[]
	for x in unique:
		left=[]
		right=[]
		for y in range(0,len(classv)):
			if feature[y]<x :
				left.append(classv[y])
			else:
				right.append(classv[y])
		frac1=float(len(left))/float(len(classv))
		frac2=float(len(right))/float(len(classv))
		entl=entropy(left)
		entr=entropy(right)
		a=ent-frac1*entl-frac2*entr
		values.append(a)
	answer.append(max(values))
#		answers.append(answer)
#for x in answer:
#	print(x)
l=[]
with open("result.csv",'r') as csvfile2:
	csvreader2=csv.reader(csvfile2)
	l=list(csvreader2)
l.append(answer)
with open("result.csv", 'w') as csvfile1: 
    csvwriter= csv.writer(csvfile1) 
    csvwriter.writerows(l)
csvfile2.close()
csvfile1.close()
csvfile.close()







