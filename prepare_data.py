from data_structure import Corpus
import argparse

import cPickle
def main():
    corpus = Corpus()
    corpus.load('Data/CQA/cqa.train', 'train')
    corpus.load('Data/CQA/cqa.dev', 'dev')
    corpus.load('Data/CQA/cqa.test', 'test')
    corpus.preprocess()
    
    options =  dict(max_answers=15, max_sents=60, max_tokens=100, skip_gram=False, emb_size=200)
    print('Start training word embeddings')
    corpus.w2v(options)

    instance, instance_dev, instance_test, embeddings, vocab = corpus.prepare(options)
    cPickle.dump((instance, instance_dev, instance_test, embeddings, vocab),open('Data/CQA.pkl','w'))


main()
