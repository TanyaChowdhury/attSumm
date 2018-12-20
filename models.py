import tensorflow as tf
from neural import dynamicBiRNN, decoding_layer, get_structure,LReLu,process_decoder_input
import numpy as np



class StructureModel():
    def __init__(self, config):
        self.config = config
        t_variables = {}
        t_variables['keep_prob'] = tf.placeholder(tf.float32)
        t_variables['batch_l'] = tf.placeholder(tf.int32)
        
        #Placeholder for answers and abstracts
        t_variables['token_idxs'] = tf.placeholder(tf.int32, [None, None, None, None])
        t_variables['abstract_idxs'] = tf.placeholder(tf.int32, [None,None,None])

        #Storing length of each heirarchy element
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None, None,None])
        t_variables['ans_l'] = tf.placeholder(tf.int32, [None, None])
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None])
        t_variables['abstract_sent_len'] = tf.placeholder(tf.int32,[None,None])
        t_variables['abstract_len'] = tf.placeholder(tf.int32,[None])

        #Storing upper limit of each element length
        t_variables['max_sent_l'] = tf.placeholder(tf.int32)
        t_variables['max_doc_l'] = tf.placeholder(tf.int32)
        t_variables['max_answers'] = tf.placeholder(tf.int32)
        t_variables['max_abstract_l'] = tf.placeholder(tf.int32)
        t_variables['max_abstract_sent_l'] = tf.placeholder(tf.int32)

        #Masks to limit element sizes
        t_variables['mask_tokens'] = tf.placeholder(tf.float32, [None, None, None,None])
        t_variables['mask_sents'] = tf.placeholder(tf.float32, [None, None,None])
        t_variables['mask_answers']= tf.placeholder(tf.float32,[None,None])
        t_variables['mask_parser_1'] = tf.placeholder(tf.float32, [None, None, None])
        t_variables['mask_parser_2'] = tf.placeholder(tf.float32, [None, None, None])
        
        self.t_variables = t_variables


    def get_feed_dict(self, batch):
        batch_size = len(batch)
        doc_l_matrix = np.zeros([batch_size], np.int32)
        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            doc_l_matrix[i] = n_sents
        
        max_doc_l = np.max(doc_l_matrix)
        max_sent_l = max([max([len(sent) for sent in doc.token_idxs]) for doc in batch])
        
        token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l], np.int32)
        sent_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
        
        abstract_idx_matrix = np.zeros([batch_size,max_doc_l,max_sent_l], np.int32)

        mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_sent_l], np.float32)
        mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float32)

        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            abstract_idx_matrix[i] = instance.abstract_idxs

            for j, sent in enumerate(instance.token_idxs):
                token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
                mask_tokens_matrix[i, j, len(sent):] = 0
                sent_l_matrix[i, j] = len(sent)
            mask_sents_matrix[i, n_sents:] = 0
        
        mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_1[:, :, 0] = 0
        mask_parser_2[:, 0, :] = 0
        
        if (self.config.large_data):
            if (batch_size * max_doc_l * max_sent_l * max_sent_l > 16 * 200000):
                return [batch_size * max_doc_l * max_sent_l * max_sent_l / (16 * 200000) + 1]

        feed_dict = {self.t_variables['token_idxs']: token_idxs_matrix, self.t_variables['sent_l']: sent_l_matrix,
                     self.t_variables['mask_tokens']: mask_tokens_matrix, self.t_variables['mask_sents']: mask_sents_matrix,
                     self.t_variables['doc_l']: doc_l_matrix, self.t_variables['abstract_idxs']: abstract_idx_matrix,
                     self.t_variables['max_sent_l']: max_sent_l, self.t_variables['max_doc_l']: max_doc_l,
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
            #Placeholders for hierarchial model at document level(structural part)
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
        abstract_sent_l = self.t_variables['abstract_sent_l']
        
        #Maximum lengths of sentences, answers and documents to be processed
        max_sent_l = self.t_variables['max_sent_l']
        max_ans_l = self.t_variables['max_doc_l']
        max_doc_l = self.t_variables['max_answers']
        max_abstract_l = self.t_variables['max_abstract_l']
        max_abstract_sent_l = self.t_variables['max_abstract_sent_l']

        #batch size
        batch_l = self.t_variables['batch_l']

        #Creating embedding matrices for answers and abstracts corresponding to indexes
        tokens_input = tf.nn.embedding_lookup(self.embeddings, self.t_variables['token_idxs'][:,:max_doc_l, :max_ans_l, :max_sent_l])
        reference_input = tf.nn.embedding_lookup(self.embeddings,self.t_variables['abstract_idxs'][:,:max_abstract_l,:max_abstract_sent_l])
        
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

        sents_output, _ = dynamicBiRNN(sents_input, ans_l, n_hidden=self.config.dim_hidden, cell_type=self.config.rnn_cell, cell_name='Model/doc')

        sents_sem = tf.concat([sents_output[0][:,:,:self.config.dim_sem], sents_output[1][:,:,:self.config.dim_sem]], 2)
        sents_str = tf.concat([sents_output[0][:,:,self.config.dim_sem:], sents_output[1][:,:,self.config.dim_sem:]], 2)

        str_scores_ = get_structure('doc', sents_str,max_ans_l, self.t_variables['mask_parser_1'], self.t_variables['mask_parser_2'])  #batch_l,  sent_l+1, sent_l
        str_scores = tf.matrix_transpose(str_scores_)  # soft parent
        sents_sem_root = tf.concat([tf.tile(embeddings_root, [batch_l, 1, 1]), sents_sem], 1)
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
        ans_output, _ = dynamicBiRNN(ans_input, doc_l, n_hidden=self.config.dim_hidden, cell_type=self.config.rnn_cell, cell_name='Model/ans')

        ans_sem = tf.concat([ans_output[0][:,:,:self.config.dim_sem], ans_output[1][:,:,:self.config.dim_sem]], 2)
        ans_str = tf.concat([ans_output[0][:,:,self.config.dim_sem:], ans_output[1][:,:,self.config.dim_sem:]], 2)

        str_scores_ = get_structure('ans', sents_str,max_doc_l, self.t_variables['mask_parser_1'], self.t_variables['mask_parser_2'])  #batch_l,  sent_l+1, sent_l
        str_scores = tf.matrix_transpose(str_scores_)  # soft parent
        ans_sem_root = tf.concat([tf.tile(embeddings_root, [batch_l, 1, 1]), sents_sem], 1)
        ans_output_ = tf.matmul(str_scores, ans_sem_root)
        ans_output = LReLu(tf.tensordot(tf.concat([ans_sem, sents_output_], 2), w_comb, [[2], [0]]) + b_comb)

        if (self.config.doc_attention == 'sum'):
            ans_output = ans_output * tf.expand_dims(mask_answers,2)
            ans_output = tf.reduce_sum(ans_output, 1)
        elif (self.config.doc_attention == 'mean'):
            ans_output = ans_output * tf.expand_dims(mask_answers,2)
            ans_output = tf.reduce_sum(ans_output, 1)/tf.expand_dims(tf.cast(doc_l,tf.float32),1)
        elif (self.config.doc_attention == 'max'):
            ans_output = ans_output + tf.expand_dims((mask_answers-1)*999,2)
            ans_output = tf.reduce_max(ans_output, 1)

        targets = self.t_variables['abstract_idxs']
        targets = tf.reshape(targets, [batch_l*, ])
        train_output, infer_output = decoding_layer(targets, ans_output, self.config)
        
        if mode == 'train' :
            decoder_output = train_output
        else:
            decoder_output = infer_output

        with tf.variable_scope('output_projection'):
            w = tf.get_variable('w', [self.config.dim_hidden, self.config.vsize], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            w_t = tf.transpose(w)
            v = tf.get_variable('v', [self.config.vsize], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
            for i,output in enumerate(decoder_output):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
            vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_n2orm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self.config.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


