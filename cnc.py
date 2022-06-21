import numpy as np
from sentence_transformers import SentenceTransformer,util
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import pandas as pd 

class EncodeSents:

    def __init__(self,sentences):
        self.__model = SentenceTransformer('model')
        self.__sentences = sentences
        self.__encodings = self.__model.encode(self.__sentences)

    def get_encodings(self):
        return self.__encodings

    def get_model(self):
        return self.__model

    def get_sentences(self):
        return self.__sentences
    
    def set_sentences(self,sentences):
        self.__sentences = sentences
        self.__init__(self.__sentences)

    def add_sentence(self,sentence):
        self.__sentences.append(sentence)
        self.__encodings = np.append(self.__encodings,[self.__model.encode(sentence)],axis=0)



class Cluster(EncodeSents):

    def __init__(self,sentences,num_of_clusters):
        super().__init__(sentences)   
        self.__num_of_clusters = num_of_clusters
        self.__sentences = sentences
        self.km = KMeans(n_clusters=num_of_clusters)
        self.__encodings = super().get_encodings()


    def get_results(self):
        self.clusters = self.km.fit(self.get_encodings())
        self.__res = pd.DataFrame({'sentence':self.__sentences,'cluster':self.clusters.labels_})
        return self.__res

    def save_results_as_csv(self,file_name):
        self.__res.to_csv(file_name)




class Classify(EncodeSents):

    def __init__(self,sentences,labels):
        assert len(sentences) == len(labels)
        super().__init__(sentences)   
        self.__labels = labels
        self.__sentences = sentences
        self.nn = MLPClassifier(max_iter=500)
        self.__encodings = super().get_encodings()


    def one_hots(self):
        label_set = set(self.__labels)
        self.__label_dict = dict()
        self.__label_vecs = []
        one_hot_vals = np.zeros((len(label_set),len(label_set)))
        for val,ohv in zip(label_set,one_hot_vals):
            ohv[val] = 1
            self.__label_dict[val] = ohv
        for label in self.__labels:
            self.__label_vecs.append(self.__label_dict[label])

        return self.__label_vecs


    def train(self):
        X = self.__encodings
        Y = self.one_hots()
        self.nn.fit(X,Y)

    def predict(self,sentence):
        enc = super().get_model().encode(sentence)
        lable_dict_rev = dict()

        pred = self.nn.predict([enc])[0]
        return list(pred).index(max(pred))

    





