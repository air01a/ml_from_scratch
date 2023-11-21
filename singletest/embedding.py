import gensim.downloader as api
import numpy as np
from os.path import isfile
from gensim.models import KeyedVectors

info = api.info()
if isfile('gtwi25.d2v'):
    model = KeyedVectors.load("gtwi25.d2v")
else:
    model = api.load("glove-twitter-25")
    model.save('gtwi25.d2v')

print(model.most_similar("pomme"))
pomme = model['pomme']
pomme_norm = np.linalg.norm(pomme)
poire = model['poire']
poire_norm = np.linalg.norm(poire)

cosine = (pomme.dot(poire)/(pomme_norm * poire_norm))
cosine_gensim = model.similarity('pomme', 'poire')

print("Dot product pomme poire: %f" % model['pomme'].dot(model['poire']))
print("Cosine similarity np: %f" % cosine )
print("Cosine similarity gensim: %f" % cosine_gensim)