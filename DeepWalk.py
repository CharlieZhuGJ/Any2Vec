# https://gitee.com/dogecheng/python/blob/master/graph/DeepWalk_and_Node2Vec.ipynb#
import networkx as nx
import sklearn
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt

# %matplotlib inline
print(gensim.__version__)
print(sklearn.__version__)
print(nx.__version__)

random_state = 2020
G = nx.read_edgelist('./wiki/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])


#####################
# DeepWalk
#####################


class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks, walkers=1, verbose=0, random_state=None):
        """
        随机游走生成节点序列，使用word2vec生成节点向量
        :param graph: networkx生成的图数据结构
        :param walk_length: 随机游走序列的长度，一般为10
        :param num_walks: 随机游走序列的数量，
        :param walkers: 线程数量
        :param verbose: 是否打印变量
        :param random_state: 随机状态
        """
        self.G = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.walkers = walkers
        self.verbose = verbose
        self.w2v = None
        self.embeddings = None
        self.random_state = (lambda x: x if x else 2020)(random_state)
        self.dataset = self.get_train_data(self.walk_length, self.num_walks, self.walkers, self.verbose)

    def fit(self, embed_size=128, window=5, n_jobs=3, epochs=5, **kwargs):
        kwargs["sentences"] = self.dataset
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = n_jobs
        kwargs["window"] = window
        kwargs["epochs"] = epochs
        kwargs["seed"] = self.random_state

        self.w2v = Word2Vec(**kwargs)

    def get_train_data(self, walk_length, num_walks, workers=1, verbose=0):
        if num_walks % workers == 0:
            num_walks = [num_walks // workers] * workers
        else:
            num_walks = [num_walks // workers] * workers + [num_walks % workers]

        nodes = list(self.G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self.simulate_walks)(nodes, num, walk_length) for num in num_walks
        )

        dataset = list(itertools.chain(*results))
        return dataset

    def simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deep_walk(walk_length=walk_length, start_node=v))
        return walks

    def deep_walk(self, walk_length, start_node):

        G = self.G

        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            current_nerghbors = list(G.neighbors(current_node))
            if len(current_nerghbors) > 0:
                walk.append(random.choice(current_nerghbors))
            else:
                break
        return walk

    def get_embeddings(self):
        if self.w2v:
            self.embeddings = {}
            for node in self.G.nodes():
                self.embeddings[node] = self.w2v.wv[node]
            return self.embeddings
        else:
            print("Please train the model first")
            return None


class Classifier(object):
    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = clf
        self.label_encoder = MultiLabelBinarizer()

    def evaluate(self, y_true, y_pred):
        average_list = ["micro", "macro", "samples", "weighted"]
        results = {}
        y_true = self.label_encoder.transform(y_true)
        y_pred = self.label_encoder.transform(y_pred)
        for average in average_list:
            results[average] = f1_score(y_true, y_pred, average=average)
        return results

    def split_train_evaluate(self, X, Y, test_size=0.2, **kwargs):
        X = [self.embeddings[x] for x in X]
        self.label_encoder.fit(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, **kwargs)
        self.clf.fit(X_train, Y_train)
        y_pred = self.clf.predict(X_test)
        return self.evaluate(Y_test, y_pred)


def evaluate_embeddings(embeddings, clf, X, Y, **kwargs):
    clf = Classifier(embeddings, clf)
    res = clf.split_train_evaluate(X, Y, **kwargs)
    return res


def plot_embeddings(X, Y, embeddings, ):
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    plt.figure(figsize=(15, 10))
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


X, Y = [], []
with open("./wiki/wiki_labels.txt", "r") as f:
    for line in f:
        x, y = line.split()
        X.append(x)
        Y.append(y)
# %time
model = DeepWalk(G, walk_length=10, num_walks=80, walkers=3, verbose=0, random_state=random_state)
model.fit(embed_size=128, window=5, n_jobs=1, epochs=3)
embeddings = model.get_embeddings()

clf = LogisticRegression(solver="liblinear", random_state=random_state)
evaluate_embeddings(embeddings, clf, X, Y, test_size=0.2, random_state=random_state)

plot_embeddings(X, Y, embeddings)

