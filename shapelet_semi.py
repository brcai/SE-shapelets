import numpy as np
from funcs import *
import random
from copy import copy
import scipy as sp
import warnings
from sklearn.cluster import SpectralClustering
import sklearn.metrics.pairwise as pdist
from sklearn.metrics import pairwise_distances_chunked
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import multiprocessing
import timeit
from sklearn import datasets, metrics

class SemiShapelet:
	def __init__(self,x_train,y_train,x_test,y_test,Y,K,eta,k,ll):
		self.x_train=x_train
		self.y_train=y_train
		self.x_test=x_test
		self.y_test=y_test
		self.Y=Y #groud truth labels
		self.K=K #number of clusters
		self.k=k #number of shapelets
		self.eta=eta #weight of variance in lda
		self.ll=ll #shapelet length
		self.feat=[] #selected shapelets
		self.rep=np.zeros([self.x_test.shape[0],self.k]) #bag-of-shapelet representation
		self.labels=[] #clusterd labels
		return

	def getOneChain(self,idx):
		ts=self.x_train[idx,:].tolist()[0]
		l=len(ts)
		ll=self.ll
		k=self.k
		cachelist=np.matrix(np.ones((l-ll+1+2,l-ll+1+2)) * np.inf)
		for j in range(1,l-ll+1+2):
			cachelist[j,0]=1
			cachelist[-1,j]=1
		cachelist[-1,-1]=np.inf
		for j in range(l-ll+1):
			for t in range(j+ll,l-ll+1):
				t1=ts[j:j+ll]
				t2=ts[t:t+ll]
				dd=eu(t1,t2)
				cachelist[t+1,j+1]=-dd
		for j in range(l-ll+1):
			cachelist[j,j]=np.inf
		maxd,path=shortest(cachelist,k+2)
		chain=[path[j]-1 for j in range(1,len(path)-1)]
		chain.append(idx)
		return chain

	def getChain(self):
		D=self.x_train
		k=self.k
		ll=self.ll
		chainidx=np.zeros([D.shape[0],k])
		chainlist=[]
		for i in range(D.shape[0]):
			onechain=self.getOneChain(i)
			for j in onechain:
				chainlist.append(D[i,j:j+ll].tolist()[0])
		#chainlist=np.mat(chainlist)
		chainmat=np.zeros([len(chainlist),len(chainlist[0])])
		for i in range(len(chainlist)):
			for j in range(len(chainlist[0])):
				chainmat[i,j]=chainlist[i][j]
		kmeans=KMeans(n_clusters=2*k, random_state=0).fit(chainlist)
		feats=[]
		for i in kmeans.cluster_centers_:
			feats.append(i)
		return feats

	def getChainPara(self):
		idx=[i for i in range(self.x_train.shape[0])]
		p = multiprocessing.Pool()
		b = p.map(self.getOneChain, idx)
		chainidx=np.array(b)
		p.close()
		p.join()
		chainlist=[]
		D=self.x_train
		ll=self.ll
		k=self.k
		for i in range(self.x_train.shape[0]):
			for j in chainidx[i][:-1]:
				chainlist.append(D[chainidx[i][-1],j:j+ll].tolist()[0])
		chainmat=np.zeros([len(chainlist),len(chainlist[0])])
		for i in range(len(chainlist)):
			for j in range(len(chainlist[0])):
				chainmat[i,j]=chainlist[i][j]
		kmeans=KMeans(n_clusters=2*k, random_state=0).fit(chainmat)
		feats=[]
		for i in kmeans.cluster_centers_:
			feats.append(i)
		return feats

	def mindist(self,x,sub):
		dist=np.inf
		for i in range(len(x)-len(sub)+1):
			tmp=eu(x[i:i+len(sub)],sub)
			if tmp<dist:
				dist=tmp
		return dist

	def lds(self,subs):
		x_rep=np.zeros([self.x_train.shape[0],len(subs)])
		for i in range(self.x_train.shape[0]):
			x=self.x_train[i,:].tolist()[0]
			for j in range(len(subs)):
				x_rep[i,j]=self.mindist(x,subs[j])
		u=x_rep.mean(axis=0)
		uc=[[] for i in range(self.K)]
		dt=[[] for i in range(self.K)]
		for i in range(self.x_train.shape[0]):
			dt[self.y_train[i]].append(x_rep[i,:].tolist())
		for i in dt:
			if len(i)<4: continue #print("skip")
		for i in range(len(dt)):
			tmp=np.mat(dt[i])
			uc[i]=tmp.mean(axis=0).tolist()[0]
		rank=[0. for i in range(len(subs))]
		for i in range(len(subs)):
			a=sum([len(dt[aa])*(uc[aa][i]-u[i])**2 for aa in range(self.K)])
			tmp=0.
			for j in range(self.x_train.shape[0]):
				tmp+=(x_rep[j,i]-uc[self.y_train[j]][i])**2
			b=tmp
			rank[i]=a-self.eta*b
		ranksort=copy(rank)
		ranksort.sort(reverse=True)
		for i in range(self.k):
			self.feat.append(subs[rank.index(ranksort[i])])
		return

	def repData(self):
		for i in range(self.x_test.shape[0]):
			for j in range(len(self.feat)):
				self.rep[i,j]=self.mindist(self.x_test[i,:].tolist()[0],self.feat[j])
		return

	def run(self):
		chainlist=self.getChainPara()
		self.lds(chainlist)
		self.repData()
		clustering=SpectralClustering(n_clusters=self.K,assign_labels="discretize",random_state=0).fit(self.rep)
		self.label=clustering.labels_
		return


if __name__ == "__main__":
	print("Running shapelet:")
	files=["BeetleFly",
	"CBF",
	"Coffee",
	"TwoLeadECG"
	]
	iddx=int(input())
	file=files[iddx]
	print('Running on: '+file)
	dataset,label,K=read(file)
	per = 0.05 #size of labelled data
	idx=[]
	num=[0 for i in range(K)]
	allnum=[0 for i in range(K)]
	for i in range(len(label)):
		allnum[label[i]]+=1
	for i in range(len(label)):
		if num[label[i]]<=allnum[label[i]]*per:
			idx.append(i)
			num[label[i]]+=1
	idx=np.array(idx)
	x_train=[]
	x_test=[]
	y_train=[]
	y_test=[]
	for i in range(len(dataset)):
		if i in idx:
			x_train.append(dataset[i])
			y_train.append(label[i])
		else:
			x_test.append(dataset[i])
			y_test.append(label[i])
	x_train=np.mat(x_train)
	x_test=np.mat(x_test)
	inst=SemiShapelet(x_train,y_train,np.mat(dataset),label,label,K,k=5,eta=0.001,ll=int(len(dataset[0])/30))
	inst.run()
	ri=rand_index_score(inst.label,inst.y_test)
	ri=round(ri,4)
	print(ri)
