from data_structure import Corpus
import argparse

import cPickle
def main():
    corpus = Corpus()
    corpus.load('Data/CQA/train.bin', 'train')
    corpus.load('Data/CQA/val.bin', 'dev')
    corpus.load('Data/CQA/test.bin', 'test')
    corpus.preprocess()
    
    options =  dict(max_sents=60, max_tokens=100, skip_gram=False, emb_size=200)
    print('Start training word embeddings')
    corpus.w2v(options)

    instance, instance_dev, instance_test, embeddings, vocab = corpus.prepare(options)
    cPickle.dump((instance, instance_dev, instance_test, embeddings, vocab),open('Data/CQA.pkl','w'))


main()