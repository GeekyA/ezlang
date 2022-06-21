# ezlang
ezlang is a python module that is supposed to make ML based NLP easy for non practitioners. 
The library is largely based on the sentence_transformers package for creating contextual embeddings(numerical representation) for sentences
these embeddings can then be used to accomplish two tasks, text clustering and classification via a very simple interface.


Classification example
```
from ezlang.cnc import Classify

c = Classify(sentences = ['hi how are you','hey, what up','stop doing that.','it is annoying'],labels = [0,0,2,1])  
c.train()
for i in ['hi how are you','hey, what up','stop doing that.','it is annoying']:
    print(c.predict(i))
```
Output
```
0
0
2
1
```


Clustering example
```
from ezlang.cnc import Classify

c = Classify(sentences = ['hi how are you','hey, what up','stop doing that.','it is annoying'],num_of_clusters = 2)
c.get_results()
```
The output for this should be a pandas dataframe with two columns 'sentence' & 'cluster' wherein the model would have successfully divide the data into
the number of clusters mentioned and the df will explain which sentence belongs to what cluster.

Pending functionality:
-Ability to save models.
-Add prediction method for clustering model
-Add semantic search functionality
