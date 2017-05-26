""" Adapted from sudeep raja's implementation """
import numpy as np
from sklearn.neighbors import BallTree, KDTree

# Base class
class LRU_KNN:
    def __init__(self, capacity, key_dimension, delta=0.001, alpha=0.1):
        self.capacity = capacity
        self.curr_capacity = 0
        self.delta = delta
        self.alpha = alpha

        self.embeddings = np.zeros((capacity, key_dimension))
        self.values = np.zeros(capacity)

        self.lru = np.zeros(capacity)
        self.tm = 0.0


    # Returns the distances and indexes of the closest embeddings
    def _nn(self, keys, k):
        pass

    # Inserts emebddings and values at the given indices
    def _insert(self, keys, values, indices):
        pass

    # Returns the stored embeddings and values of the closest embeddings
    def query(self, keys, k):
        _, indices = self._nn(keys, k)
        
        embs = [] ; values = []
        for ind in indices:
          self.lru[ind] = self.tm
          
          embs.append(self.embeddings[ind])
          values.append(self.values[ind])

        self.tm+=0.01

        return embs, values

    # Adds new embeddings (and values) to the dictionary
    def add(self, keys, values):

        skip_indices = []
        if self.curr_capacity >= 1:
            dist, ind = self._nn(keys, k=1)
            for i, d in enumerate(dist):
                if d[0] < self.delta:
                    index = ind[i][0]
                    new_value = values[i]
                    self.values[index] = self.values[index]*(1-self.alpha) + new_value*self.alpha
                    skip_indices.append(i)

        indices, keys_, values_ = [], [], []
        for i, _ in enumerate(keys):
            if i in skip_indices: continue
            if self.curr_capacity >= self.capacity:
                # find the LRU entry
                index = np.argmin(self.lru)
            else:
                index = self.curr_capacity
                self.curr_capacity+=1
            self.lru[index] = self.tm
            indices.append(index) ; keys_.append(keys[i]) ; values_.append(values[i])

        self._insert(keys_, values_, indices)
        self.tm += 0.01



# Simple KD-tree dict
class kdtree_dict(LRU_KNN):
    def __init__(self, capacity, key_dimension, delta=0.001, alpha=0.1):
        LRU_KNN.__init__(self, capacity, key_dimension, delta, alpha)
        self.tree = None

    def _nn(self, keys, k):
        dist, ind = self.tree.query(keys, k=k)
        return dist, ind

    def _insert(self, keys, values, indices):
        for i, ind in enumerate(indices):
            #print "num: " + str(i) + ", index: " + str(ind)
            self.embeddings[ind] = keys[i]
            self.values[ind] = values[i]
        self.tree = KDTree(self.embeddings[:self.curr_capacity])



class q_dictionary:
    def __init__(self, capacity, key_dimension, num_actions, delta=0.001, alpha=0.1):
        self.num_actions = num_actions
        self.delta = delta
        self.alpha = alpha
        self.dicts = []

        for a in xrange(num_actions):
            new_dict = kdtree_dict(capacity, key_dimension, self.delta, self.alpha)
            self.dicts.append(new_dict)

    def _query(self, embeddings, actions, knn):
        # Return the embeddings and values of nearest neighbours from the DNDs for the given embeddings and actions
        dnd_embeddings = [] ; dnd_values = []
        for i, a in enumerate(actions):
            e, v = self.dicts[a].query([embeddings[i]], knn)
            dnd_embeddings.append(e[0]) ; dnd_values.append(v[0])
        return dnd_embeddings, dnd_values


    def query(self, embeddings, knn):
        # Return the embeddings and values of nearest neighbours from the DNDs of each action forthe given embeddings
        dnd_embeddings = [] ; dnd_values = []
        for i, a in enumerate(actions):
            e, v = self.dicts[a].query(embeddings[i], knn)
            dnd_embeddings.append(e[0]) ; dnd_values.append(v[0])
        # Return format is action# x batch# x embedding
        return dnd_embeddings, dnd_values


    def add(self, embeddings, actions, values):
        # Adds the given embeddings to the corresponding dicts
        for a in range(self.num_actions):
          e = [] ; v = []
          for i, _ in enumerate(embeddings):
            if actions[i] == a:
              e.append(embeddings[i]) ; v.append(values[i])
 
          if e:
            self.dicts[a].add(e, v)
        return True

    def curr_capacity(self):
        min_val = 0
        for a in range(self.num_actions):
            min_val = min(min_val, self.dicts[a].curr_capacity)
        return min_val
