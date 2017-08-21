""" Adapted from sudeep raja's implementation """
from collections import OrderedDict
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

        #(Least returned unit counter could probably be wrapped into its own class)
        self.lru = np.zeros(capacity)
        self.tm = 0.0


    # Returns the distances and indexes of the closest embeddings
    def _nn(self, keys, k):
        pass

    # Inserts emebddings and values at the given indices
    def _insert(self, keys, values, indices):
        pass

    def queryable(self, k):
        return self.curr_capacity > k

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
        #Doesn't seem to work properly, and design is cleaner (and faster without)
        if False: #self.queryable(1):
            dist, ind = self._nn(keys, k=1)
            for i, d in enumerate(dist):
                index = ind[i][0]
                new_emb = keys[i]
                retrieved_emb = self.embeddings[index]
                if np.allclose(new_emb, retrieved_emb, atol=self.delta, rtol=0.0):
                    new_value = values[i]
                    self.values[index] = self.values[index]*(1-self.alpha) + new_value*self.alpha
                    skip_indices.append(i)


        indices, keys_, values_ = [], [], []
        for i, _ in enumerate(keys):
            if i in skip_indices:
                continue
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
        
    def save(self, filename):
        save_data = [ self.embeddings,
                      self.values,
                      self.curr_capacity,
                      self.lru,
                      self.tm ]
        np.save(filename, save_data)
    
    def load(self, filename):
        save_data = np.load(filename)
        self.embeddings, self.values, self.curr_capacity, self.lru, self.tm = save_data
        self._rebuild_index()


# Dict using ANNOY library for approximate KNN. Rebuilds the tree every n units added
# Could probably be a bit more efficient
#TODO: search through cache to find nn's too
class annoy_dict(LRU_KNN):
    def __init__(self, capacity, key_dimension, delta=0.001, alpha=0.1, batch_size=100):
        LRU_KNN.__init__(self, capacity, key_dimension, delta, alpha)

        from annoy import AnnoyIndex
        self.index = AnnoyIndex(key_dimension, metric='euclidean')
        self.index.set_seed(123)

        self.initial_update_size = batch_size
        self.min_update_size = self.initial_update_size
        self.cached_keys = []
        self.cached_values = []
        self.cached_indices = []

        self.built_capacity = 0

    def _nn(self, keys, k):
        dists = [] ; inds = []
        for key in keys:
            ind, dist = self.index.get_nns_by_vector(key, k, include_distances=True)
            dists.append(dist) ; inds.append(ind)
        return dists, inds

    def _insert(self, keys, values, indices):
        self.cached_keys = self.cached_keys + keys
        self.cached_values = self.cached_values + values
        self.cached_indices = self.cached_indices + indices

        if len(self.cached_indices) >= self.min_update_size:
            self.min_update_size = max(self.initial_update_size, self.curr_capacity*0.02)
            self._update_index()

    def _update_index(self):
        self.index.unbuild()
        for i, ind in enumerate(self.cached_indices):
            new_key = self.cached_keys[i]
            new_value = self.cached_values[i]
            self.embeddings[ind] = new_key
            self.values[ind] = new_value
            self.index.add_item(ind, new_key)

        self.cached_keys = []
        self.cached_values = []
        self.cached_indices = []

        self.index.build(50)
        self.built_capacity = self.curr_capacity
        
    def _rebuild_index(self):
        self.index.unbuild()
        for ind, emb in enumerate(self.embeddings[:self.curr_capacity]):
            self.index.add_item(ind, emb)
        self.index.build(50)
        self.built_capacity = self.curr_capacity

    def queryable(self, k):
        return (LRU_KNN.queryable(self, k) and (self.built_capacity > k))


# Simple KD-tree dict
class kdtree_dict(LRU_KNN):
    def __init__(self, capacity, key_dimension, delta=0.001, alpha=0.1):
        LRU_KNN.__init__(self, capacity, key_dimension, delta, alpha)
        self.tree = None
        self.built_capacity = 0

    def _nn(self, keys, k):
        dists, inds = self.tree.query(keys, k=k)
        return dists, inds

    def _insert(self, keys, values, indices):
        for i, ind in enumerate(indices):
            #print "num: " + str(i) + ", index: " + str(ind)
            self.embeddings[ind] = keys[i]
            self.values[ind] = values[i]
        self._rebuild_index()
            
    def _rebuild_index(self):
        self.tree = KDTree(self.embeddings[:self.curr_capacity])
        self.built_capacity = self.curr_capacity

    def queryable(self, k):
        return (LRU_KNN.queryable(self, k) and (self.built_capacity > k))


class q_dictionary:
    def __init__(self, capacity, key_dimension, num_actions, delta=0.001, alpha=0.1):
        self.num_actions = num_actions
        self.delta = delta
        self.alpha = alpha
        self.dicts = []

        for a in xrange(num_actions):
            new_dict = annoy_dict(capacity, key_dimension, self.delta, self.alpha)
            self.dicts.append(new_dict)

    def _query(self, embeddings, actions, knn):
        # Return the embeddings and values of nearest neighbours from the
        #   DNDs for the given embeddings and actions
        # This could probably be made more efficient by batching the querys by actions
        dnd_embeddings = [] ; dnd_values = []
        for i, a in enumerate(actions):
            e, v = self.dicts[a].query([embeddings[i]], knn)
            dnd_embeddings.append(e[0]) ; dnd_values.append(v[0])
        # Return format is batch# x dist# x embedding
        return dnd_embeddings, dnd_values


    def query(self, embeddings, knn):
        # Return the embeddings and values of nearest neighbours from the
        #   DNDs of each action forthe given embeddings
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

    def min_capacity(self):
        min_val = 0
        for a in range(self.num_actions):
            min_val = min(min_val, self.dicts[a].curr_capacity)
        return min_val

    def tot_capacity(self):
        tot_val = 0
        for a in range(self.num_actions):
            tot_val += self.dicts[a].curr_capacity
        return tot_val


    def queryable(self, k):
        for a in range(self.num_actions):
            if not self.dicts[a].queryable(k): return False

        return True
        
    def save(self, name):
        for a in range(self.num_actions):
            self.dicts[a].save(name + '_dict_' + str(a) + '.npy')
        
    def load(self, name):
        for a in range(self.num_actions):
            self.dicts[a].load(name + '_dict_' + str(a) + '.npy')

class alpha_KNN:
    def __init__(self, capacity, key_dimension, delta=0.001, alpha=0.1, batch_size=1000):
        self.capacity = capacity
        self.curr_capacity = 0
        self.delta = delta
        self.alpha = 0.001

        self.embeddings = np.zeros((capacity, key_dimension))
        self.values = np.zeros(capacity)

        self.weights = np.zeros(capacity)

        from annoy import AnnoyIndex
        self.index = AnnoyIndex(key_dimension, metric='euclidean')
        self.index.set_seed(123)

        self.min_update_size = batch_size
        self.cached_keys = []
        self.cached_values = []
        self.cached_indices = []

        self.built_capacity = 0

    def _nn(self, keys, k):
        dists = [] ; inds = []
        for key in keys:
            ind, dist = self.index.get_nns_by_vector(key + [1.0], k, include_distances=True)
            dists.append(dist) ; inds.append(ind)
        return dists, inds

    def _insert(self, keys, values, indices):
        self.cached_keys = self.cached_keys + keys
        self.cached_values = self.cached_values + values
        self.cached_indices = self.cached_indices + indices

        if len(self.cached_indices) >= self.min_update_size:
            self._rebuild_index()

    def _rebuild_index(self):
        self.index.unbuild()
        for i, ind in enumerate(self.cached_indices):
            new_key = self.cached_keys[i]
            new_value = self.cached_values[i]
            self.embeddings[ind] = new_key
            self.values[ind] = new_value
            self.weights[ind] = new_weight
            self.index.add_item(ind, new_key + [new_weight])

        self.cached_keys = []
        self.cached_values = []
        self.cached_indices = []

        self.index.build(50)
        self.built_capacity = self.curr_capacity

    def queryable(self, k):
        return (self.built_capacity > k)

    # Returns the stored embeddings and values of the closest embeddings
    def query(self, keys, k):
        _, indices = self._nn(keys, k)
        
        embs = [] ; values = [] ; weights = []
        for ind in indices:
          embs.append(self.embeddings[ind])
          values.append(self.values[ind])
          weights.append(self.weights[ind])

        return embs, values, weights

    # Adds new embeddings (and values) to the dictionary
    def add(self, keys, values):

        if self.queryable(5):
            dists, inds = self._nn(keys, k=5)
            for ind, dist in enumerate(dists):
                for i, d in enumerate(dist):
                    index = inds[ind][i]
                    self.weights[index] *= (1-self.alpha)

        indices, keys_, values_ = [], [], []
        for i, _ in enumerate(keys):
            if self.curr_capacity >= self.capacity:
                # find the LRU entry
                index = np.argmin(self.weights)
            else:
                index = self.curr_capacity
                self.curr_capacity+=1
            self.weights[index] = 1.0
            indices.append(index) ; keys_.append(keys[i]) ; values_.append(values[i])

        self._insert(keys_, values_, indices)
