


import re, string
from nltk.corpus import stopwords
from string import punctuation
import copy




regex = re.compile('[%s]' % re.escape(string.punctuation))
cachedStopWords = stopwords.words('english')

class kMeans():
    def __init__(self, seeds, tweets):
        self.__seeds = seeds
        self.__tweets = tweets
        self.__max_iterations = 1000
        self.__k = len(seeds)

        self.clusters = {} # cluster to tweetID
        self.__rev_clusters = {} # reverse index, tweetID to cluster
        self.jaccardMatrix = {} # stores pairwise jaccard distance in a matrix

        self.__initializeClusters()
        self.__initializeMatrix()
        
    def jaccardDistance(self, setA, setB):
        # Calcualtes the Jaccard Distance of two sets
        try:
            return 1 - float(len(setA.intersection(setB))) / float(len(setA.union(setB)))
        except TypeError:
            print ('Invalid type. Type set expected.')
    def bagOfWords(self, string):
        # Returns a bag of words from a given string
        # Space delimited, removes punctuation, lowercase
        # Cleans text from url, stop words, tweet @, and 'rt'
        words = string.lower().strip().split(' ')
        for word in words:
            word = word.rstrip().lstrip()
            if not re.match(r'^https?:\/\/.*[\r\n]*', word) \
            and not re.match('^@.*', word) \
            and not re.match('\s', word) \
            and word not in cachedStopWords \
            and word != 'rt' \
            and word != '':
                yield regex.sub('', word)
                
    def __initializeMatrix(self):
        # Dynamic Programming: creates matrix storing pairwise jaccard distances
        for ID1 in self.__tweets:
            self.jaccardMatrix[ID1] = {}
            bag1 = set(self.bagOfWords(self.__tweets[ID1]))
            for ID2 in self.__tweets:
                if ID2 not in self.jaccardMatrix:
                    self.jaccardMatrix[ID2] = {}
                bag2 = set(self.bagOfWords(self.__tweets[ID2]))
                distance = self.jaccardDistance(bag1, bag2)
                self.jaccardMatrix[ID1][ID2] = distance
                self.jaccardMatrix[ID2][ID1] = distance
                
    def __initializeClusters(self):
        # Initialize tweets to no cluster
        for ID in self.__tweets:
            self.__rev_clusters[ID] = -1

        # Initialize clusters with seeds
        for k in range(self.__k):
            self.clusters[k] = set([self.__seeds[k]])
            self.__rev_clusters[self.__seeds[k]] = k
            
    def calcNewClusters(self):
        # Initialize new cluster
        new_clusters = {}
        new_rev_cluster = {}
        loss_total=0
        for k in range(self.__k):
            new_clusters[k] = set()

        for ID in self.__tweets:
            min_dist = float("inf")
            min_dist1 = float("inf")
            min_cluster = self.__rev_clusters[ID]

            # Calculate min average distance to each cluster
            cnt=0
            for k in self.clusters:
                dist = 0
                count = 0
                cnt+=1
                for ID2 in self.clusters[k]:
                    dist += self.jaccardMatrix[ID][ID2]
                    count += 1
                if count > 0:
                    avg_dist = dist/float(count)
                    if min_dist > avg_dist:
                        min_dist = avg_dist
                        min_cluster = k
                        
            for k in self.clusters:
                if(k==min_cluster):
                    continue
                dist = 0
                count = 0
                cnt+=1
                for ID2 in self.clusters[k]:
                    dist += self.jaccardMatrix[ID][ID2]
                    count += 1
                if count > 0:
                    avg_dist1 = dist/float(count)
                    if min_dist1 > avg_dist1:
                        min_dist1 = avg_dist1
                        min_prev = k
                        
                        
                        
            a = 0
            b = 0
            count = 0  

            
            for ID2 in self.clusters[min_cluster]:
                a += self.jaccardMatrix[ID][ID2]
                count +=1
            
            for ID2 in self.clusters[min_prev]:
                b += self.jaccardMatrix[ID][ID2]

            if(count==1):
                loss_total+=0
            else:
                
                loss_total+=(b-a)/max(a,b)
            new_clusters[min_cluster].add(ID)
            new_rev_cluster[ID] = min_cluster
        
        loss_total=loss_total/len(self.__tweets)
        
        return new_clusters, new_rev_cluster, loss_total
    
    
    def converge(self):
        # Initialize previous cluster to compare changes with new clustering
        new_clusters, new_rev_clusters, _ = self.calcNewClusters()
        self.clusters = copy.deepcopy(new_clusters)
        self.__rev_clusters = copy.deepcopy(new_rev_clusters)

        # Converges until old and new iterations are the same
        iterations = 1
        while iterations < self.__max_iterations:
            new_clusters, new_rev_clusters,_ = self.calcNewClusters()
            iterations += 1
            if self.__rev_clusters != new_rev_clusters:
                self.clusters = copy.deepcopy(new_clusters)
                self.__rev_clusters = copy.deepcopy(new_rev_clusters)
            else:
                #print iterations
                return
            
    def printClusterText(self):
        # Prints text of clusters
        for k in self.clusters:
            for ID in self.clusters[k]:
                print(self.__tweets[ID])
            print ('\n')
 
    def printClusters(self):
        # Prints cluster ID and tweet IDs for that cluster
        for k in self.clusters:
            print (str(k) + ':' + ','.join(map(str,self.clusters[k])))

    def printMatrix(self):
        # Prints jaccard distance matrix
        for ID in self.__tweets:
            for ID2 in self.__tweets:
                print (ID, ID2, self.jaccardMatrix[ID][ID2])
                
