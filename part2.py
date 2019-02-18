from textblob import TextBlob as tb
from nltk.corpus import stopwords
import random
import time
import pandas as pd

#Open file with utf encoding
train = pd.read_csv("train_original.csv",encoding='utf-8')

#Fill spaces with null
train = train.fillna('null')

#Number of hash functions
numHashes=50

ids={}

docNames=[]


#Convert a document into a set of shingles
def generateShingle(doc,wordSet):
    doc=tb(doc)
    words=""
    #Remove stopwords from doc
    for word in doc.words:
        if word not in stopwords.words('english'):
            words+=word
    #Create shingles from the new doc, convert them into integers and add 
    #them to the given set
    j=0
    while j<len(words)-5:
        #create shingle of 5 characters
        shingle=words[j:j+5]
        newShingle=hash(shingle) & 0xffffffff
        #newShingle = bin(int(hashlib.md5(shingle.encode('utf-8')).hexdigest(),16))
        wordSet.add(newShingle)
        j+=1
        
t1 = time.time()
i=0
for question in train.question1:
    docNames.append(question)
    wordSet=set()
    generateShingle(question,wordSet)
    ids[question]=wordSet
    i+=2

i=1
for question in train.question2:
    docNames.append(question)
    wordSet=set()
    generateShingle(question,wordSet)
    ids[question]=wordSet
    i+=2
numDocs=i-2

maxShingleID = 2**32-1
def pickRandomCoeffs(k):
  # Create a list of 'k' random values.
    randList = []
  
    while k > 0:
    # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID) 
  
    # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID) 
    
    # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1
    
    return randList

coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

signatures = []
nextPrime = 4294967311


# Rather than generating a random permutation of all possible shingles, 
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index 
# of the first shingle that you would have encountered in the random order.

# For each document...
for docID in docNames:
  
    # Get the shingle set for this document.
    shingleIDSet = ids[docID]
  
    # The resulting minhash signature for this document. 
    signature = []
    
    # For each of the random hash functions...
    for i in range(0, numHashes):
    
        # For each of the shingles actually in the document, calculate its hash code
        # using hash function 'i'. 
    
        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
        # the maximum possible value output by the hash.
        minHashCode = nextPrime + 1
    
        # For each shingle in the document...
        for shingleID in shingleIDSet:
            # Evaluate the hash function.
            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime
      
            # Track the lowest hash code seen.
            if hashCode < minHashCode:
                minHashCode = hashCode
                
        # Add the smallest hash code value as component number 'i' of the signature.
        signature.append(minHashCode)
    # Store the MinHash signature for this document.
    signatures.append(signature)
    
#Divide a list of signatures in bands. Hash every signature in the range of the
#band. If 2 signatures hash in the same bucket then they are a candidate pair.
def LSH(signatures, bands):
    t0 = time.time()
    #Calcucate the number of rows of every band
    rows=numHashes/bands
    duplicates=set() #contains the near duplicates
    band=0
    #For every band
    while band<bands:
        #initialize the bucket of the band
        bucket={}
        i=0
        #For every signature
        for signature in signatures:
            #Find the position of the row in the signature
            row=rows*band
            #Create a list of the minhash values
            values=[]
            #For every row in the range of this band
            while row<rows*(band+1):
                values.append(signature[row])
                row+=1
            value=tuple(values)
            if value not in bucket:
                signs=[] #signatures that are similar
                sig=i
                signs.append(sig)
                bucket[value]=signs
            else:
                signs=[]
                signs=bucket[value]
                sig=i
                signs.append(sig)
                bucket[value]=signs
            i+=1
        #if 'null' in 
        #duplicate.remove('null')
        for key in bucket:
            list=bucket[key]
            i=0
            while(i<len(list)-1):
                j=i+1
                while(j<len(list)):
                    duplicate=[list[i],list[j]]
                   # set(duplicate=tuple(d))
                    duplicates.add(tuple(duplicate))
                    j+=1
                i+=1
        band+=1
        
    elapsed = (time.time() - t0)
    print "\nFinding candidates took %.2fsec" % elapsed 
    return duplicates
elapsed = (time.time() - t1)
print "\nCalcucating MinHash signatures took %.2fsec" % elapsed
def computeJaccard(duplicates):
    t3 = time.time()
    for duplicate in duplicates:
        sig1=ids[docNames[duplicate[0]]]
        sig2=ids[docNames[duplicate[1]]]
        if len(sig1)!=0:
            #print docNames[duplicate[0]],docNames[duplicate[1]], sig1, sig2
            J = (len(sig1.intersection(sig2)) / float(len(sig1.union(sig2))))
            if J>0.8 :#threshold
                print "\n %5s --> %5s  Jaccard:%.2f" % (docNames[duplicate[0]], docNames[duplicate[1]], J)
    elapsed = (time.time() - t3)
    print "\n Computing Jaccard similarities took %.2fsec" % elapsed

duplicates=LSH(signatures,6)
computeJaccard(duplicates)
