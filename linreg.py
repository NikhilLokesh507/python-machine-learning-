import csv
import math
import statistics


def RMSE(list1,c):
	sum1=0
	for x in range(0,len(list1)):
		sum1+=(list1[x]-c[x])**2
	val=math.sqrt(sum1/len(list1))
	return val


def MMRE(list1,c):
	sum1=0
	for x in range(0,len(list1)):
		sum1+=(math.fabs(list1[x]-c[x]))/math.fabs(c[x]+0.005)
	return sum1/len(list1)

def Covar(list1,list2):
	m1=statistics.mean(list1)
	m2=statistics.mean(list2)
	sum1=0
	for x in range(0,len(list1)):
		sum1+=(list1[x]-m1)*(list2[x]-m2)
	return sum1/(len(list1)-1)

def PearCorr(list1,list2):
	m1=statistics.mean(list1)
	m2=statistics.mean(list2)
	sum1=0
	sum2=0
	sum3=0
	for x in range(0,len(list1)):
		sum1+=(list1[x]-m1)*(list2[x]-m2)
	
	for x in range(0,len(list1)):
		sum2+=(list1[x]-m1)**2
	for x in range(0,len(list2)):
		sum3+=(list2[x]-m2)**2
	denominator=math.sqrt(sum2*sum3)
	return sum1/denominator

def Normalize(l):
	hi=max(l)
	lo=min(l)
	if hi==lo:
		l=["NA"]
	else:
		l=[(x-lo)/(hi-lo) for x in l]
	#print(l)
	return l


def LinReg(c,feature):
	#c=Normalize(c)
	feature=Normalize(feature)
	if feature[0]=="NA":
		return ["NA"]
	l=[]
	p=(PearCorr(c,feature))
	#print(p)
	p=math.fabs(p)
	if p>=0.7:
		b1=Covar(c,feature)*statistics.stdev(c)/statistics.stdev(feature)
		l.append(b1)
		b0=statistics.mean(c)-b1*statistics.mean(feature)
		l.append(b0)
	else:
		l.append("NA")
	return l



filename="37.csv"
rows = []
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader:
        rows.append([float(col) for col in row])  
features=[]
numf=len(rows[0])
numr=len(rows)
for x in range(0,numf):
	feature=[]
	for row in rows:
		feature.append(row[x])
	features.append(feature)
	classv=[]
#classv.append(features[numf-1])
for x in features[numf-1]:
	classv.append(x)
classv=Normalize(classv)
answer=[]
rmse=[]
mmre=[]
ho=int(0.7*numr)
#ho is the magnitude of the holdout, we are using holdout method.
for feature in features:
	answer.append(LinReg(classv[:ho],feature[:ho]))
	#we are modelling for the SLR model for objects with indices from 0 to ho
for feature in features:
	ind=features.index(feature)
	if answer[ind][0]!="NA":
		l=[answer[ind][0]*feature[x]+answer[ind][1] for x in range(ho,numr)]
           #l is the list that contains predicted values if the class variable.
		rmse.append(RMSE(l,classv[ho:]))
		mmre.append(MMRE(l,classv[ho:]))
	else:
		rmse.append("NA")
		mmre.append("NA")

print(answer)
'''with open("result.csv",'r') as csvfile2:
    csvreader2=csv.reader(csvfile2)
    l=list(csvreader2)
l.append(answer)
with open("result.csv", 'w') as csvfile1: 
    csvwriter= csv.writer(csvfile1) 
    csvwriter.writerows(l)
csvfile2.close()
csvfile1.close()
csvfile.close()'''
