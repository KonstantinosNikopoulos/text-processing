from textblob import TextBlob as tb
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

train = pd.read_csv("train_original.csv",encoding='utf-8')

#Fill spaces with null
train = train.fillna('null')

ids={}

docNames=[]

#Update an inverted index
def updateIndex(index,doc,docID):
    doc=tb(doc)
    for word in doc.words:
        if word in stopwords.words('english'):
            doc.words.remove(word)
    for word in doc.words:
        if word not in index:
            docs=[1,[]] #signatures that are similar
            docs[1].append(tuple((docID,doc.words.count(word))))
            index[word]=docs
        else:
            docs=[] #signatures that are similar
            docs=index[word]
            docs[0]+=1
            docs[1].append(tuple((docID,doc.words.count(word))))
            index[word]=docs
    

invertedIndex={}
docs=[]
docNames=[]
ids={}
i=0
for question in train.question1:
    docNames.append(question)
    ids[i]=question
    updateIndex(invertedIndex,question,i)
    if i==1000:
        break
    i+=2

i=1
for question in train.question2:
    docNames.append(question)
    ids[i]=question
    updateIndex(invertedIndex,question,i)
    if i==1001:
        break
    i+=2

N=i-2

def find_top_k(map,k):
    top=[]
    j=0
    while j<k:
        top.append(-1)
        max=-100
        for element in map:
            if map[element]>max:
                max=map[element]
                top[j]=element
        map.pop(top[j], None)
        j+=1
    return top

def findSimilar(query,index):
    query=tb(query)
    acc={}
    for word in query.words:
        if word in index:
            nt=index[word][0]
            idf=np.log(1+N/nt)
            docs=index[word][1]
            for doc in docs:
                tf=1+np.log([doc[1]])
                if doc in acc:
                    acc[doc[0]]+=tf*idf
                else:
                    acc[doc[0]]=tf*idf
    top_k=find_top_k(acc,10)
    return top_k

printed=0
stop=1
while stop!=0:
    query=input("Give query:")
    for top in findSimilar(query,invertedIndex):
        if top!=-1:
            print docNames[top]
            printed=1
    if printed==0:
        print "Similar queries couldn't be found"
    stop=input("Press anything to give next query or press 0 to stop:")