import tensorflow as tf

def LReLu(x, leak=0.01):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


def dynamicBiRNN(input, seqlen, n_hidden, cell_type, cell_name=''):
    batch_size = tf.shape(input)[0]
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32):
        if(cell_type == 'gru'):
            fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        elif(cell_type == 'lstm'):
            fw_cell = tf.contrib.rnn.LSTMCell(n_hidden)

        fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name + 'bw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32):
        if(cell_type == 'gru'):
            bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        elif(cell_type == 'lstm'):
            bw_cell = tf.contrib.rnn.LSTMCell(n_hidden)
        bw_initial_state = bw_cell.zero_state(batch_size, tf.float32)
    
    with tf.variable_scope(cell_name):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 sequence_length=seqlen)
    return outputs, output_states


def decoder(input, seqlen, n_hidden):
    batch_size = tf.shape(input)[0]
    with tf.variable_scope('decoder_cell', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32):
        decoder_cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True, initializer=self.rand_unif_init)

    decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('decoder'):
        outputs, output_states = tf.nn.dynamic_rnn(decoder_cell, input,
                                                                 initial_state=decoder_initial_state,
                                                                 sequence_length=seqlen)


    return outputs, output_state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,target_sequence_length, max_summary_length,output_layer, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,target_sequence_length)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,helper,encoder_state,output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_summary_length)
    return outputs

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
   
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,output_keep_prob=keep_prob)
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], start_of_sequence_id), end_of_sequence_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,helper,encoder_state,output_layer)
    
    outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,maximum_iterations=max_target_sequence_length)
    return outputs

def decoding_layer(dec_input, encoder_state,config):
    
    target_sequence_length = target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_sequence_length = 100

    target_vocab_size = config.vsize
    decoding_embedding_size = config.dim_hidden

    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.LSTMCell(config.dim_hidden,state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
    
    keep_prob = config.keep_prob
    batch_size = config.batch_size

    target_vocab_to_int = config.vocab

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state,cells,dec_embed_input,target_sequence_length,max_target_sequence_length, 
                                            output_layer,keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state,cells,dec_embeddings,target_vocab_to_int['<GO>'],target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length,target_vocab_size,output_layer,batch_size,keep_prob)

    return (train_output, infer_output)


def get_structure(name, input, max_l, mask_parser_1, mask_parser_2):
    def _getDep(input, mask1, mask2):
        #input: batch_l, sent_l, rnn_size
        with tf.variable_scope("Structure/"+name, reuse=True, dtype=tf.float32):
            w_parser_p = tf.get_variable("w_parser_p")
            w_parser_c = tf.get_variable("w_parser_c")
            b_parser_p = tf.get_variable("bias_parser_p")
            b_parser_c = tf.get_variable("bias_parser_c")

            w_parser_s = tf.get_variable("w_parser_s")
            w_parser_root = tf.get_variable("w_parser_root")

        parent = tf.tanh(tf.tensordot(input, w_parser_p, [[2], [0]]) + b_parser_p)
        child = tf.tanh(tf.tensordot(input, w_parser_c, [[2], [0]])+b_parser_c)
        # rep = LReLu(parent+child)
        temp = tf.tensordot(parent,w_parser_s,[[-1],[0]])
        raw_scores_words_ = tf.matmul(temp,tf.matrix_transpose(child))

        # raw_scores_words_ = tf.squeeze(tf.tensordot(rep, w_parser_s, [[3], [0]]) , [3])
        raw_scores_root_ = tf.squeeze(tf.tensordot(input, w_parser_root, [[2], [0]]) , [2])
        raw_scores_words = tf.exp(raw_scores_words_)
        raw_scores_root = tf.exp(raw_scores_root_)
        tmp = tf.zeros_like(raw_scores_words[:,:,0])
        raw_scores_words = tf.matrix_set_diag(raw_scores_words,tmp)

        str_scores, LL = _getMatrixTree(raw_scores_root, raw_scores_words, mask1, mask2)
        return str_scores

    def _getMatrixTree(r, A, mask1, mask2):
        L = tf.reduce_sum(A, 1)
        L = tf.matrix_diag(L)
        L = L - A
        LL = L[:, 1:, :]
        LL = tf.concat([tf.expand_dims(r, [1]), LL], 1)
        LL_inv = tf.matrix_inverse(LL)  #batch_l, doc_l, doc_l
        d0 = tf.multiply(r, LL_inv[:, :, 0])
        LL_inv_diag = tf.expand_dims(tf.matrix_diag_part(LL_inv), 2)
        tmp1 = tf.matrix_transpose(tf.multiply(tf.matrix_transpose(A), LL_inv_diag))
        tmp2 = tf.multiply(A, tf.matrix_transpose(LL_inv))
        d = mask1 * tmp1 - mask2 * tmp2
        d = tf.concat([tf.expand_dims(d0,[1]), d], 1)
        return d, LL

    str_scores = _getDep(input, mask_parser_1, mask_parser_2)
    return str_scores