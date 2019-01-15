import tensorflow as tf
from neural import dynamicBiRNN, get_structure,LReLu,decode
import numpy as np
from tensorflow.python.layers.core import Dense

class StructureModel():
    def __init__(self, config):
        self.config = config
        t_variables = {}
        t_variables['keep_prob'] = tf.placeholder(tf.float32)
        t_variables['batch_l'] = tf.placeholder(tf.int32)
        
        #Placeholder for answers and abstracts
        t_variables['token_idxs'] = tf.placeholder(tf.int32, [None, None, None, None])
        t_variables['abstract_idxs'] = tf.placeholder(tf.int32, [None,None])
        t_variables['generated_idxs'] = tf.placeholder(tf.int32,[None,None])

        #Storing length of each heirarchy element
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None, None,None])
        t_variables['ans_l'] = tf.placeholder(tf.int32, [None, None])
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None])
        t_variables['abstract_l'] = tf.placeholder(tf.int32,[None])

        #Storing upper limit of each element length
        t_variables['max_sent_l'] = tf.placeholder(tf.int32)
        t_variables['max_doc_l'] = tf.placeholder(tf.int32)
        t_variables['max_ans_l'] = tf.placeholder(tf.int32)
        t_variables['max_abstract_l'] = tf.placeholder(tf.int32)

        #Masks to limit element sizes
        t_variables['mask_tokens'] = tf.placeholder(tf.float32, [None, None, None,None])
        t_variables['mask_sents'] = tf.placeholder(tf.float32, [None, None,None])
        t_variables['mask_answers']= tf.placeholder(tf.float32,[None,None])
        
        #Parser Masks
        t_variables['mask_parser_1'] = tf.placeholder(tf.float32, [None, None, None])
        t_variables['mask_parser_2'] = tf.placeholder(tf.float32, [None, None, None])

        t_variables['start_tokens'] = tf.placeholder(tf.int32,[None])

        
        self.t_variables = t_variables


    def get_feed_dict(self, batch):
        batch_size = len(batch)
        abstracts_l_matrix = np.zeros([batch_size],np.int32)
        doc_l_matrix = np.zeros([batch_size], np.int32)

        for i, instance in enumerate(batch):
            n_ans = len(instance.token_idxs)
            n_words = len(instance.abstract_idxs)
            doc_l_matrix[i] = n_ans
            abstracts_l_matrix[i] = n_words
        
        max_doc_l = np.max(doc_l_matrix)
        max_ans_l = max([max([len(ans) for ans in doc.token_idxs]) for doc in batch])
        max_sent_l = max([max([max([len(sent) for itr in doc.token_idxs for sent in itr]) for ans in doc.token_idxs]) for doc in batch])
        max_abstract_l = np.max(abstracts_l_matrix)

        ans_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
        sent_l_matrix = np.zeros([batch_size, max_doc_l, max_ans_l], np.int32)

        token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_ans_l, max_sent_l], np.int32)
        abstract_idx_matrix = np.zeros([batch_size,max_abstract_l], np.int32)

        mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_ans_l, max_sent_l], np.float32)
        mask_sents_matrix = np.ones([batch_size, max_doc_l, max_ans_l], np.float32)
        mask_answers_matrix = np.ones([batch_size, max_doc_l],np.float32)
        mask_abstact_matrix = np.ones([batch_size,max_abstract_l],np.float32)

        for i, instance in enumerate(batch):
            n_answers = len(instance.token_idxs)
            abstract_ = instance.abstract_idxs
            abstract_idx_matrix[i,:len(abstract_)] = np.asarray(abstract_)
            mask_abstact_matrix[i,len(abstract_):] = 0
            abstracts_l_matrix[i] = len(abstract_)

            for j, ans in enumerate(instance.token_idxs):
                for k, sent in enumerate(instance.token_idxs[j]):
                    token_idxs_matrix[i, j, k,:len(sent)] = np.asarray(sent)
                    mask_tokens_matrix[i, j, k,len(sent):] = 0
                    sent_l_matrix[i, j,k] = len(sent)

                mask_sents_matrix[i,j,len(ans):]=0
                ans_l_matrix[i,j] = len(ans)

            mask_answers_matrix[i, n_answers:] = 0
        
        mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_1[:, :, 0] = 0
        mask_parser_2[:, 0, :] = 0
        
        feed_dict = {self.t_variables['token_idxs']: token_idxs_matrix,self.t_variables['abstract_idxs']: abstract_idx_matrix,
                     self.t_variables['sent_l']: sent_l_matrix,self.t_variables['ans_l']:ans_l_matrix,self.t_variables['doc_l']: doc_l_matrix, 
                     self.t_variables['mask_tokens']: mask_tokens_matrix, self.t_variables['mask_sents']: mask_sents_matrix, self.t_variables['mask_answers']:mask_answers_matrix,
                     self.t_variables['abstract_l']:abstracts_l_matrix,
                     self.t_variables['max_sent_l']: max_sent_l,self.t_variables['max_ans_l']:max_ans_l, self.t_variables['max_doc_l']: max_doc_l,self.t_variables['max_abstract_l']: max_abstract_l,
                     self.t_variables['mask_parser_1']: mask_parser_1, self.t_variables['mask_parser_2']: mask_parser_2,
                     self.t_variables['batch_l']: batch_size, self.t_variables['keep_prob']:self.config.keep_prob}
        
        return  feed_dict



    def build(self):
        with tf.variable_scope("Embeddings"):
            #Initial embedding placeholders
            self.embeddings = tf.get_variable("emb", [self.config.n_embed, self.config.d_embed], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root = tf.get_variable("emb_root", [1, 1, 2 * self.config.dim_sem], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root_a = tf.get_variable("emb_root_ans", [1, 1,2* self.config.dim_sem], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root_s = tf.get_variable("emb_root_s", [1, 1,2* self.config.dim_sem], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Model"):
            #Weights and biases at pooling layers and final softmax for output. (Fianl might not be required)(Semantic combination part)
            w_comb = tf.get_variable("w_comb", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb = tf.get_variable("bias_comb", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_comb_a = tf.get_variable("w_comb_a", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb_a = tf.get_variable("bias_comb_a", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_comb_s = tf.get_variable("w_comb_s", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb_s = tf.get_variable("bias_comb_s", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_softmax = tf.get_variable("w_softmax", [2 * self.config.dim_sem, self.config.dim_output], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_softmax = tf.get_variable("bias_softmax", [self.config.dim_output], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Structure/doc"):
            #Placeholders for hierarchical model at document level(structural part)
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Structure/ans"):
            #Placeholders for  hierarchial model at answer level(structural part)
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Structure/sent"):
            #Placeholders for hierarchial model at sentence level(structural part)
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        #Variables of dimension batchsize passing length of each vector to architectures
        sent_l = self.t_variables['sent_l']
        ans_l = self.t_variables['ans_l']
        doc_l = self.t_variables['doc_l']
        abstract_l = self.t_variables['abstract_l']
        
        #Maximum lengths of sentences, answers and documents to be processed
        max_sent_l = self.t_variables['max_sent_l']
        max_ans_l = self.t_variables['max_ans_l']
        max_doc_l = self.t_variables['max_doc_l']
        max_abstract_l = self.t_variables['max_abstract_l']

        #batch size
        batch_l = self.t_variables['batch_l']

        #Creating embedding matrices for answers and abstracts corresponding to indexes
        tokens_input = tf.nn.embedding_lookup(self.embeddings, self.t_variables['token_idxs'][:,:max_doc_l, :max_ans_l, :max_sent_l])
        reference_input = tf.nn.embedding_lookup(self.embeddings,self.t_variables['abstract_idxs'][:,:max_abstract_l])
        
        #Dropout on input
        tokens_input = tf.nn.dropout(tokens_input, self.t_variables['keep_prob'])

        #Masking inputs
        mask_tokens = self.t_variables['mask_tokens'][:,:max_doc_l, :max_ans_l, :max_sent_l]
        mask_sents = self.t_variables['mask_sents'][:, :max_doc_l,:max_ans_l]
        mask_answers = self.t_variables['mask_answers'][:,:max_doc_l]


        [_, _, _, _, rnn_size] = tokens_input.get_shape().as_list()
        tokens_input_do = tf.reshape(tokens_input, [batch_l * max_doc_l*max_ans_l, max_sent_l, rnn_size])

        sent_l = tf.reshape(sent_l, [batch_l * max_doc_l* max_ans_l])
        mask_tokens = tf.reshape(mask_tokens, [batch_l * max_doc_l*max_ans_l, -1])

        #Word level input
        tokens_output, _ = dynamicBiRNN(tokens_input_do, sent_l, n_hidden=self.config.dim_hidden,
                                        cell_type=self.config.rnn_cell, cell_name='Model/sent')
        
        tokens_sem = tf.concat([tokens_output[0][:,:,:self.config.dim_sem], tokens_output[1][:,:,:self.config.dim_sem]], 2)
        tokens_str = tf.concat([tokens_output[0][:,:,self.config.dim_sem:], tokens_output[1][:,:,self.config.dim_sem:]], 2)
        
        temp1 = tf.zeros([batch_l * max_doc_l*max_ans_l, max_sent_l,1], tf.float32)
        temp2 = tf.zeros([batch_l * max_doc_l*max_ans_l ,1,max_sent_l], tf.float32)

        mask1 = tf.ones([batch_l * max_doc_l * max_ans_l, max_sent_l, max_sent_l-1], tf.float32)
        mask2 = tf.ones([batch_l * max_doc_l * max_ans_l, max_sent_l-1, max_sent_l], tf.float32)
        
        mask1 = tf.concat([temp1,mask1],2)
        mask2 = tf.concat([temp2,mask2],1)

        str_scores_s_ = get_structure('sent', tokens_str, max_sent_l, mask1, mask2)  # batch_l,  sent_l+1, sent_l
        str_scores_s = tf.matrix_transpose(str_scores_s_)  # soft parent
        tokens_sem_root = tf.concat([tf.tile(embeddings_root_s, [batch_l * max_doc_l *max_ans_l, 1, 1]), tokens_sem], 1)
        tokens_output_ = tf.matmul(str_scores_s, tokens_sem_root)
        tokens_output = LReLu(tf.tensordot(tf.concat([tokens_sem, tokens_output_], 2), w_comb_s, [[2], [0]]) + b_comb_s)

        if (self.config.sent_attention == 'sum'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)
        elif (self.config.sent_attention == 'mean'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)/tf.expand_dims(tf.cast(sent_l,tf.float32),1)
        elif (self.config.sent_attention == 'max'):
            tokens_output = tokens_output + tf.expand_dims((mask_tokens-1)*999,2)
            tokens_output = tf.reduce_max(tokens_output, 1)

        #Sentence level RNN
        sents_input = tf.reshape(tokens_output, [batch_l*max_doc_l, max_ans_l,2*self.config.dim_sem])
        ans_l = tf.reshape(ans_l,[batch_l*max_doc_l])

        sents_output, _ = dynamicBiRNN(sents_input, ans_l, n_hidden=self.config.dim_hidden, cell_type=self.config.rnn_cell, cell_name='Model/ans')

        sents_sem = tf.concat([sents_output[0][:,:,:self.config.dim_sem], sents_output[1][:,:,:self.config.dim_sem]], 2)
        sents_str = tf.concat([sents_output[0][:,:,self.config.dim_sem:], sents_output[1][:,:,self.config.dim_sem:]], 2)

        temp1 = tf.zeros([batch_l * max_doc_l, max_ans_l, 1], tf.float32)
        temp2 = tf.zeros([batch_l * max_doc_l, 1, max_ans_l], tf.float32)

        mask1 = tf.ones([batch_l * max_doc_l , max_ans_l, max_ans_l-1], tf.float32)
        mask2 = tf.ones([batch_l * max_doc_l , max_ans_l-1, max_ans_l], tf.float32)
        
        mask1 = tf.concat([temp1,mask1],2)
        mask2 = tf.concat([temp2,mask2],1)

        str_scores_ = get_structure('ans', sents_str,max_ans_l, self.t_variables['mask_parser_1'], self.t_variables['mask_parser_2'])  #batch_l,  sent_l+1, sent_l
        str_scores = tf.matrix_transpose(str_scores_)  # soft parent
        sents_sem_root = tf.concat([tf.tile(embeddings_root_a, [batch_l*max_doc_l, 1, 1]), sents_sem], 1)
        sents_output_ = tf.matmul(str_scores, sents_sem_root)
        sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_output_], 2), w_comb, [[2], [0]]) + b_comb)

        if (self.config.doc_attention == 'sum'):
            sents_output = sents_output * tf.expand_dims(mask_sents,2)
            sents_output = tf.reduce_sum(sents_output, 1)
        elif (self.config.doc_attention == 'mean'):
            sents_output = sents_output * tf.expand_dims(mask_sents,2)
            sents_output = tf.reduce_sum(sents_output, 1)/tf.expand_dims(tf.cast(ans_l,tf.float32),1)
        elif (self.config.doc_attention == 'max'):
            sents_output = sents_output + tf.expand_dims((mask_sents-1)*999,2)
            sents_output = tf.reduce_max(sents_output, 1)

        #Answer level RNN
        ans_input = tf.reshape(sents_output, [batch_l, max_doc_l,2*self.config.dim_sem])
        ans_output, answer_states = dynamicBiRNN(ans_input, doc_l, n_hidden=self.config.dim_hidden, cell_type=self.config.rnn_cell, cell_name='Model/doc')

        ans_sem = tf.concat([ans_output[0][:,:,:self.config.dim_sem], ans_output[1][:,:,:self.config.dim_sem]], 2)
        ans_str = tf.concat([ans_output[0][:,:,self.config.dim_sem:], ans_output[1][:,:,self.config.dim_sem:]], 2)

        str_scores_ = get_structure('doc', ans_str, max_doc_l, self.t_variables['mask_parser_1'], self.t_variables['mask_parser_2'])  #batch_l,  sent_l+1, sent_l
        str_scores = tf.matrix_transpose(str_scores_)  # soft parent
        ans_sem_root = tf.concat([tf.tile(embeddings_root, [batch_l, 1, 1]), sents_sem], 1)
        ans_output_ = tf.matmul(str_scores, ans_sem_root)
        ans_output = LReLu(tf.tensordot(tf.concat([ans_sem, sents_output_], 2), w_comb, [[2], [0]]) + b_comb)

        print tf.shape(ans_output)
        # if (self.config.doc_attention == 'sum'):
        #     ans_output = ans_output * tf.expand_dims(mask_answers,2)
        #     ans_output = tf.reduce_sum(ans_output, 1)
        # elif (self.config.doc_attention == 'mean'):
        #     ans_output = ans_output * tf.expand_dims(mask_answers,2)
        #     ans_output = tf.reduce_sum(ans_output, 1)/tf.expand_dims(tf.cast(doc_l,tf.float32),1)
        # elif (self.config.doc_attention == 'max'):
        #     ans_output = ans_output + tf.expand_dims((mask_answers-1)*999,2)
        #     ans_output = tf.reduce_max(ans_output, 1)



        tgt_vocab_size = self.config.vsize
        learning_rate = self.config.lr
        
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.dim_hidden)
        helper = tf.contrib.seq2seq.TrainingHelper(ans_output, abstract_l, time_major=True)
        projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,tf.contrib.rnn.LSTMStateTuple(tf.random_normal([batch_l,self.config.dim_hidden]),tf.random_normal([batch_l,self.config.dim_hidden])),output_layer=projection_layer)
        outputs, states,seq_l = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.t_variables['abstract_idxs'][:,:max_abstract_l], logits=logits)

        target_weights = tf.sequence_mask(abstract_l,dtype=tf.float32)
        product = crossent*target_weights
        loss_numerator =  tf.reduce_sum(product)
        train_loss = (loss_numerator /tf.to_float(batch_l) )

        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        self.final_output = logits
        self.loss = train_loss
        self.opt = optimizer.minimize(train_loss)

