import gensim
import numpy as np
import re
import random
import math
import unicodedata
import itertools
from utils import grouper
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', unicode(s,'utf-8'))
                  if unicodedata.category(c) != 'Mn')

class RawData:
    def __init__(self,name,count):
        self.idx = name+ '/'+str(count)
        self.abstract = []
        self.answers = []
        self.prediction = ''
        
class DataSet:
    def __init__(self, data):
        self.data = data
        self.num_examples = len(self.data)

    def sort(self):
        random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x._max_sent_len)
        self.data = sorted(self.data, key=lambda x: x._doc_len)

    def get_by_idxs(self, idxs):
        return [self.data[idx] for idx in idxs]

    def get_batches(self, batch_size, num_epochs=None, rand = True):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        idxs = list(range(self.num_examples))
        _grouped = lambda: list(grouper(idxs, batch_size))

        if(rand):
            grouped = lambda: random.sample(_grouped(), num_batches_per_epoch)
        else:
            grouped = _grouped
        num_steps = num_epochs*num_batches_per_epoch
        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for i in range(num_steps):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            yield i,batch_data


class Instance:
    def __init__(self):
        self.token_idxs = None
        self.abstract_idxs = None
        self.idx = -1

    def _doc_len(self, idx):
        k = len(self.token_idxs)
        return k

    def _abstract_len(self,idx):
        k = len(self.abstract_idxs)
        return k

    def _max_sent_len(self, idxs):
        k = max([len(sent) for sent in self.token_idxs])
        return k

class Corpus:
    def __init__(self):
        self.doclst = {}

    def load(self, in_path, name):
        self.doclst[name] = []        
        binaryData = []

        for item in open(in_path):
            binaryData.append(item)
        
        i = 0
        while(i< len(binaryData)):
            doc = RawData(name,int(i/3))

            #Check if everything all right
            split1 = binaryData[i]
            if ( i!=0 and '<split1>' not in split1):
                print 'Some error in preprocess', i,"\n"
                print split1

            #Create Answer List
            answersString = binaryData[i+1]
            answersString = answersString.replace('<0>','')
            answersString = answersString.replace('\n','')
            answersList = answersString.split('<split2>')
            answersList = [item.split('<split3>') for item in answersList]
            doc.answers = answersList

            #Create Abstract List
            abstractString = binaryData[i+2]
            abstractString = abstractString.replace('<1>','')
            abstractString = abstractString.replace('\n','')
            abstractList = abstractString.split('<reference_split>')
            doc.abstract = abstractList

            i+=3

            self.doclst[name].append(doc)

    def preprocess(self):
        for dataset in self.doclst:
            for doc in self.doclst[dataset]:

                #Cleaning and tokenizing document 
                doc.answers_sent_list = []
                doc.document_tokens_list = []
                for answers in doc.answers:
                    preprocessed_sentences = []
                    token_sentences = []
                    for sentences in answers:
                        s = re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sentences)
                        preprocessed_sentences.append(s)
                        sentence_tokens = s.split()
                        if(len(sentence_tokens)>1):
                            token_sentences.append(s.split())
                    doc.answers_sent_list.append(preprocessed_sentences)
                    doc.document_tokens_list.append(token_sentences)

                #Cleaning and tokenizing abstract
                doc.abstract_sent_list = []
                doc.abstract_tokens_list = []
                for sentences in doc.abstract:
                    s = re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sentences)
                    doc.abstract_sent_list.append(s)
                    abstract_tokens = s.split()
                    if(len(abstract_tokens)>1):
                        doc.abstract_tokens_list.append(abstract_tokens)
            
            #Only add threads with more than 0 answers and more than 0 words abstract
            self.doclst[dataset] = [doc for doc in self.doclst[dataset] if (len(doc.document_tokens_list)!=0 and len(doc.abstract_tokens_list)!=0)]




    def w2v(self, options):
        sentences = []
        for doc in self.doclst['train']:
            for answers in doc.document_tokens_list:
                for sents in answers:
                    sentences.extend(sents)
            
            for sents in doc.abstract_tokens_list:     
                sentences.extend(sents)
        
        if('dev' in self.doclst):
            for doc in self.doclst['dev']:
                for answers in doc.document_tokens_list:
                    for sents in answers:
                        sentences.extend(sents)
            
                for sents in doc.abstract_tokens_list:     
                    sentences.extend(sents)
        
        if(options['skip_gram']):
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4, sg=1)
        else:
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4)
        
        self.w2v_model.scan_vocab(sentences)  # initial survey
        rtn = self.w2v_model.scale_vocab(dry_run = True)  # trim by min_count & precalculate downsampling
        print(rtn)
        self.w2v_model.finalize_vocab()  # build tables & arrays
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        self.vocab = self.w2v_model.wv.vocab
        print('Vocab size: {}'.format(len(self.vocab)))
    
    def prepare(self, options):
        instances, instances_dev, instances_test = [],[],[]
        instances, embeddings, vocab = self.prepareData(options,'train')
        
        if ('dev' in self.doclst):
            instances_dev = self.prepareData(options, 'dev')
        
        instances_test = self.prepareData(options, 'test')
        return instances, instances_dev, instances_test, embeddings, vocab

    def prepareData(self, options,mode):
        instancelst = []

        if(mode=='train'):        
            #(50000,200) every word in vocab is assigned an embedding which is pre trained
            embeddings = np.zeros([len(self.vocab)+1,options['emb_size']])
            for word in self.vocab:
                embeddings[self.vocab[word].index] = self.w2v_model[word]
            
            self.vocab['UNK'] = gensim.models.word2vec.Vocab(count=0, index=len(self.vocab))
        

        n_filtered = 0
        
        for i_doc, doc in enumerate(self.doclst[mode]):
            instance = Instance()
            instance.idx = i_doc

            n_answers = len(doc.document_tokens_list)
            max_n_sents = max([len(answer) for answer in doc.document_tokens_list])
            max_n_tokens = max([len(sent) for answer in doc.document_tokens_list for sent in answer])

            if(n_answers > options['max_answers']):
                n_filtered+=1
                continue

            if(n_sents>options['max_sents']):
                n_filtered += 1
                continue
            
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue

            #Generating document token indexes array and storing them in token_idxs of instance
            document_token_indexes = []
            for answer in doc.document_tokens_list:
                sentence_indexes = []
                for sentence in answer:
                    token_indexes = []
                    for token in sentence:
                        if(token in self.vocab):
                            token_indexes.append(self.vocab[token].index)
                        else:
                            token_indexes.append(self.vocab['UNK'].index)
                    sentence_indexes.append(token_indexes)
                document_token_indexes.append(sentence_indexes)
            instance.token_idxs = document_token_indexes


            #Generating abstract token indexes array and storing them in abstract_idxs of instance
            abstract_token_indexes = []
            for sentences in doc.abstract_tokens_list:
                token_indexes = []
                for token in sentences:
                    if(token in self.vocab):
                        token_indexes.append(self.vocab[token].index)
                    else:
                        token_indexes.append(self.vocab['UNK'].index)
                abstract_token_indexes.append(token_indexes)
            instance.abstract_idxs = abstract_token_indexes

            instancelst.append(instance)

        print('n_filtered in train: {}'.format(n_filtered))
        
        if mode == 'train':
            return instancelst, embeddings, self.vocab
        else:
            return instancelst
