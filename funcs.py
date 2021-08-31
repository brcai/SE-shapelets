import numpy as np
from decimal import Decimal
from sklearn.preprocessing import StandardScaler
from copy import copy
import pandas as pd
import re
from scipy.special import comb 
from sklearn import metrics

def shortest(G,k):
	x=0
	y=G.shape[0]-1
	D=np.zeros([y+1,k])
	P=np.zeros([y+1,k])
	for i in range(D.shape[0]):
		D[i,0]=np.inf
	D[0,0]=0
	for i in range(1,k):
		for j in range(0,len(G)):
			alld=[D[t,i-1]+G[j,t] for t in range(y+1)]
			tmp=np.min(alld)
			loc=np.argmin(alld)
			D[j,i]=tmp
			P[j,i]=loc
	path=[]
	v=y
	for i in range(1,k):
		path.append(v)
		v=int(P[v,k-i])
	path.append(x)
	path.reverse()
	return D[y,k-1],path


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def metric(y,Y):
	ri=rand_index_score(Y, y)
	nmi=metrics.normalized_mutual_info_score(Y, y)
	ari=metrics.cluster.adjusted_rand_score(Y, y)
	return ri, nmi, ari

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def readL1(file):
	#df = pd.read_csv('/home/bcai/work/shapelet/databk/'+file+'/l1.csv')
	df = pd.read_csv('D:/work/shapelet/dt/'+file+'/l1.csv')
	distMat = df.values
	return distMat

def read(file):
	#fp = open('/home/bcai/work/shapelet/databk/'+file+'/data.txt')
	fp = open('D:/work/shapelet/dt/'+file+'/data.txt')
	features = []
	label = []
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split(',', raw)
		oneRow = [itm for itm in raw if itm != '']
		label.append(int(oneRow[0]))
		features.append([float(itm) for itm in oneRow[1:]])
	tmp = list(set(label))
	k = len(tmp)
	m = {tmp[i]:i for i in range(k)}
	newlabel = [m[i] for i in label]
	fp.close()
	return features, newlabel, k

def eu(a,b):
	d=0.
	for i in range(len(a)):
		d+=(a[i]-b[i])**2
	return np.sqrt(d)


def isSingular(m):
	res = np.linalg.det(m)
	if res<=0.: return True
	return False


if __name__ == "__main__":
	files=["50words",
"CBF",
"DiatomSizeReduction",
"DistalPhalanxOutlineAgeGroup",
"DistalPhalanxTW",
"ECG5000",
"FaceFour",
"FaceAll",
"FISH",
"InsectWingbeatSound",
"Lighting7",
"MALLAT",
"Meat",
"MiddlePhalanxTW",
"NonInvasiveFatalECG_Thorax1",
"NonInvasiveFatalECG_Thorax2",
"OliveOil",
"Plane",
"ProximalPhalanxOutlineAgeGroup",
"ProximalPhalanxTW",
"ShapesAll",
"SonyAIBORobotSurface",
"StarLightCurves",
"Symbols",
"synthetic_control",
"Trace",
"Two_Patterns",
"UWaveGestureLibraryAll",
"uWaveGestureLibrary_X",
"uWaveGestureLibrary_Z",
"WordsSynonyms"]
	for file in files:
		print(file)
		#normalize(file)
		readcsv(file)
		exit()
		'''
		x,y,k=readts(file)
		num={i:0 for i in set(y)}
		for i in range(len(x)):
			num[y[i]]+=1
		print("	true clusters: ",[num[i] for i in num])
		'''