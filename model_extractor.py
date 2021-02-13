
"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

from tensorflow.python.ops import nn_ops
import common, math 
tf_float64=tf.float32
np_float64=np.float32
FLAGS = tf.app.flags.FLAGS

class SummarizationModel_GCN(object):

  def __init__(self, hps, word_vocab, entity_vocab):
    self._hps = hps
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps
#
    # encoder part
    self._data_name_batch = tf.placeholder(tf.int32, [hps.batch_size,20], name='data_name_batch')
    
    self._summary = tf.placeholder(tf_float64, [hps.batch_size, None], name='summary')
    
    self._max_sent_num = tf.placeholder(tf.int32, name='max_sent_num')
    self._max_sent_len = tf.placeholder(tf.int32, name='max_sent_len')
    
    self._max_entity_num = tf.placeholder(tf.int32, name='max_entity_num')
    self._max_mention_len = tf.placeholder(tf.int32, name='max_mention_len')
    
    self._sent_batch = tf.placeholder(tf.int32, [hps.batch_size, None, None], name='sent_batch')
    self._sent_nums = tf.placeholder(tf.int32, [hps.batch_size], name='sent_nums')
    self._sent_lens = tf.placeholder(tf.int32, [hps.batch_size, None], name='sent_lens')
    self._sent_padding_mask = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='sent_padding_mask')
    
    self._entity_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='entity_batch') 
    self._entity_nums = tf.placeholder(tf.int32, [hps.batch_size], name='entity_nums')
    self._mention_lens = tf.placeholder(tf.int32, [hps.batch_size, None], name='mention_lens')
    self._mention_batch = tf.placeholder(tf.int32, [hps.batch_size, None, None], name='mention_batch')
    self._mention_mask = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='mention_mask')
    
    self._sent_relas = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='sent_relas') #np.zeros((hps.batch_size, max_sent_num, max_sent_num), dtype=np.int32)
    self._sent_relas_DAD = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='sent_relas_DAD')
    
    self._entities2sents = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='entities2sents') #np.zeros((hps.batch_size, max_entity_num, max_sent_num), sent_relas=np.int32)
    self._entities2sents_DAD = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='entities2sents_DAD')
    self._sents2entities_DAD = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='sents2entities_DAD')
    
    self._entities2entities = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='entities2entities') #np.zeros((hps.batch_size, max_entity_num, max_entity_num), dtype=np.int32)
    self._entities2entities_DAD = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='entities2entities_DAD')

    self._pos_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='pos_batch')
    self._pos_mask = tf.placeholder(tf_float64, [hps.batch_size, None], name='pos_mask')
    self._entity_mask = tf.placeholder(tf_float64, [hps.batch_size, None], name='entity_mask')
    self._entities2entities_mask= tf.placeholder(tf_float64, [hps.batch_size, None], name='entities2entities_mask')
    
    self._sent_label_batch = tf.placeholder(tf.float32, [hps.batch_size, None], name='sent_label_batch')
    self._entity_label_batch = tf.placeholder(tf.float32, [hps.batch_size, None], name='entity_label_batch')
    
    if hps.rl_lambda > 0:
        self._rl_sent_label_batch = tf.placeholder(tf.float32, [hps.batch_size, None], name='rl_sent_label_batch')
        self._rl_entity_label_batch = tf.placeholder(tf.float32, [hps.batch_size, None], name='rl_entity_label_batch')
        self._rl_reward_batch = tf.placeholder(tf.float32, [hps.batch_size], name='rl_reward_batch')
    

  def _make_feed_dict(self, batch, just_enc=False, use_rl=True):

    feed_dict = {}
    feed_dict[self._data_name_batch] = batch.data_name_batch
    feed_dict[self._max_sent_num] = batch.max_sent_num
    feed_dict[self._max_sent_len] = batch.max_sent_len
    
    feed_dict[self._max_entity_num] = batch.max_entity_num
    feed_dict[self._max_mention_len] = batch.max_mention_len
    
    feed_dict[self._sent_batch] = batch.sent_batch
    feed_dict[self._sent_nums] = batch.sent_nums
    feed_dict[self._sent_lens] = batch.sent_lens
    feed_dict[self._sent_padding_mask] = batch.sent_padding_mask
    
    feed_dict[self._entity_batch] = batch.entity_batch
    feed_dict[self._entity_nums] = batch.entity_nums
    feed_dict[self._mention_lens] = batch.mention_lens
    feed_dict[self._mention_batch] = batch.mention_batch
    feed_dict[self._mention_mask] = batch.mention_mask
    
    feed_dict[self._sent_relas] = batch.sent_relas
    feed_dict[self._sent_relas_DAD] = batch.sent_relas_DAD
    
    feed_dict[self._entities2sents] = batch.entities2sents
    feed_dict[self._entities2sents_DAD] = batch.entities2sents_DAD
    feed_dict[self._sents2entities_DAD] = batch.sents2entities_DAD
    
    feed_dict[self._entities2entities] = batch.entities2entities
    feed_dict[self._entities2entities_DAD] = batch.entities2entities_DAD
    
    feed_dict[self._pos_batch] = batch.pos_batch
    feed_dict[self._pos_mask] = batch.pos_mask
    feed_dict[self._entity_mask] = batch.entity_mask
    feed_dict[self._entities2entities_mask] = batch.entities2entities_mask 
    
    feed_dict[self._sent_label_batch] = batch.sent_label_batch
    feed_dict[self._entity_label_batch] = batch.entity_label_batch
    
    if self._hps.rl_lambda > 0 and use_rl:
        feed_dict[self._rl_sent_label_batch] = batch.rl_sent_label_batch
        feed_dict[self._rl_entity_label_batch] = batch.rl_entity_label_batch
        if self._hps.mode != "decode":
            feed_dict[self._rl_reward_batch] = batch.rl_reward_batch
    
    return feed_dict

  def _get_rnn_cell(self, cell_class,
                    hidden_dim,
                    num_layers=1,
                    dropout_input_keep_prob=0.8,
                    dropout_output_keep_prob=1.0):
    if cell_class == "GRU":
      cell = tf.contrib.rnn.GRUCell(hidden_dim, 
                                     kernel_initializer=self.rand_unif_init, 
                                     bias_initializer=self.rand_unif_init)
    elif cell_class == "LSTM":
      cell = tf.contrib.rnn.LSTMCell(hidden_dim, 
                                     initializer=self.rand_unif_init, 
                                     state_is_tuple=False)
  
    if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell=cell,
          input_keep_prob=dropout_input_keep_prob,
          output_keep_prob=dropout_output_keep_prob)

    return cell

  def _add_sents_encoder(self, sent_inputs, sent_nums, sent_lens):
    hps = self._hps
    
    with tf.variable_scope('sents_encoder'): 
      cell_fw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2) 
      cell_bw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2) 
      
      sent_inputs = tf.reshape(sent_inputs, [-1, self._max_sent_len, hps.word_emb_dim])
      sent_lens = tf.reshape(sent_lens, [-1])
      
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sent_inputs, dtype=tf_float64, 
                                                                          sequence_length=sent_lens, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) 
      
      encoder_outputs = tf.reshape(encoder_outputs, [hps.batch_size, self._max_sent_num, self._max_sent_len, hps.hidden_dim])
      
      final_encs = self._reduce_states(fw_st, bw_st, 1, "text")
      final_encs = tf.reshape(final_encs, [hps.batch_size, self._max_sent_num, hps.hidden_dim]) 
      
      return final_encs
      

  def _add_sents_pos_emb(self, sent_encoding, pos_embbeding):
    hps = self._hps
    with tf.variable_scope('sents_encoder_with_pos'): 
      if hps.pos_emb_type == "rnn":
          cell_fw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2) 
          cell_bw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2)
          (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sent_encoding, dtype=tf_float64, 
                                                                          sequence_length=self._sent_nums, swap_memory=True)
          sent_enc = tf.concat(axis=2, values=encoder_outputs) 
          
      elif hps.pos_emb_type == "pos_emb":
          emb_sent_pos = tf.nn.embedding_lookup(pos_embbeding, self._pos_batch)
          sent_enc = sent_encoding + emb_sent_pos
        
    return sent_enc

  def _reduce_states(self, fw_st, bw_st, concat_axis=1, scope="text"):
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('sents_final_st_'+scope):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim, hidden_dim], dtype=tf_float64, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim, hidden_dim], dtype=tf_float64, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf_float64, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf_float64, initializer=self.trunc_norm_init)

      # Apply linear layer
      #old_c = tf.concat(axis=1, values=[fw_st, bw_st]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=concat_axis, values=[fw_st, bw_st]) # Concatenation of fw and bw state
      #new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return new_h # Return new cell and state

  def _add_entity_encoder(self, entity_inputs, mention_inputs):
    hps = self._hps
    
    with tf.variable_scope('mention_encoder'): 
        cell_fw = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim/2) 
        cell_bw = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim/2) 
          
        mention_inputs = tf.reshape(mention_inputs, [-1, self._max_mention_len, hps.word_emb_dim])
        mention_lens = tf.reshape(self._mention_lens, [-1])
          
        (_, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, mention_inputs, dtype=tf_float64, 
                                                                              sequence_length=mention_lens, swap_memory=True)
        #encoder_outputs = tf.concat(axis=2, values=encoder_outputs) 
          
        #encoder_outputs = tf.reshape(encoder_outputs, [hps.batch_size, self._max_entity_num, self._max_mention_len, hps.hidden_dim])
          
        final_encs = self._reduce_states(fw_st, bw_st, 1, "entity")
        final_encs = tf.reshape(final_encs, [hps.batch_size, self._max_entity_num, hps.hidden_dim]) 
        
        if hps.use_entity_embedding:
            final_encs = tf.concat([final_encs, entity_inputs], 2)
        
    #with tf.variable_scope('mention_encoder_pos'): 
    #    cell_fw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2) 
    #    cell_bw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2)
    #    (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, final_encs, dtype=tf_float64, 
    #                                                                      sequence_length=self._sent_nums, swap_memory=True)
    #    entity_enc = tf.concat(axis=2, values=encoder_outputs) 
    
    return final_encs

  def _add_entities2entities_loss(self, emb_entity_inputs, entities2entities):
      
    MIN_INV = 1e-10
    n=self._max_entity_num
    masks=tf.expand_dims(self._entities2entities_mask,2) 
    masks2=tf.matmul(masks, masks, transpose_a=False, transpose_b=True)
    emb_entity_inputs=emb_entity_inputs*masks
    
    emb_entity_inputs=tf.matmul(emb_entity_inputs, emb_entity_inputs, transpose_a=False, transpose_b=True)
    eye=tf.eye(n) 
    emb_entity_inputs_diag=tf.matrix_inverse(tf.expand_dims(eye,0)*(emb_entity_inputs+MIN_INV))
    emb_entity_inputs=emb_entity_inputs*tf.expand_dims(1-eye,0) 
    entities2entities2=tf.matmul(tf.matmul(emb_entity_inputs_diag,emb_entity_inputs),emb_entity_inputs_diag) 
    entities2entities2=tf.reshape(entities2entities2,[self._hps.batch_size,-1])
    sum=tf.expand_dims(tf.reduce_sum(entities2entities2, 1), 1)+MIN_INV
    entities2entities2=entities2entities2/sum
    
    masks2=masks2*tf.expand_dims(1-eye,0)
    masks2=tf.reshape(masks2,[self._hps.batch_size,-1])
    entities2entities2=entities2entities2*masks2+tf.float32.min*(1 - masks2)
    #entities2entities2=-tf.log(entities2entities2+MIN_INV)
    
    #entities2entities=entities2entities*tf.expand_dims(1-eye,0) 
    entities2entities=tf.reshape(entities2entities, [self._hps.batch_size, -1])
    sum=tf.expand_dims(tf.reduce_sum(entities2entities, 1), 1)+MIN_INV
    entities2entities=entities2entities/sum
    
    #entities2entities_losses=tf.reduce_sum(entities2entities*entities2entities2, 1) 
    entities2entities_losses=tf.nn.softmax_cross_entropy_with_logits(labels=entities2entities, logits=entities2entities2) 
    entities2entities_loss=tf.reduce_mean(entities2entities_losses) 
        
    return entities2entities_loss
  
  def _add_gcn(self):
      hps = self._hps
      
      activation_fn=None
      
      sent_enc = tf.contrib.layers.fully_connected(
                inputs=self._sent_enc,
                num_outputs=hps.hidden_dim,
                activation_fn=None,
                biases_initializer=self.trunc_norm_init,
                scope="sent_enc")
      
      entity_enc = tf.contrib.layers.fully_connected(
                    inputs=self._entity_enc,
                    num_outputs=hps.hidden_dim,
                    activation_fn=None,
                    biases_initializer=self.trunc_norm_init,
                    scope="entity_enc")
      
      for _ in range(hps.gcn_level):
          if hps.use_gcn_entity:
              sent_ss_enc = tf.contrib.layers.fully_connected(
                    inputs=sent_enc,
                    num_outputs=hps.hidden_dim,
                    activation_fn=activation_fn,
                    biases_initializer=self.trunc_norm_init,
                    scope="sents_ss_enc", reuse=tf.AUTO_REUSE)
              new_sent_ss_enc = tf.matmul(self._sent_relas_DAD, sent_ss_enc)
              del sent_ss_enc 
          
          if hps.use_gcn_entity:
              entity_ee_enc = tf.contrib.layers.fully_connected(
                    inputs=entity_enc,
                    num_outputs=hps.hidden_dim,
                    activation_fn=activation_fn,
                    biases_initializer=self.trunc_norm_init,
                    scope="entity_ee_enc", reuse=tf.AUTO_REUSE)
              new_entity_ee_enc = tf.matmul(self._entities2entities_DAD, entity_ee_enc)
              del entity_ee_enc 
          
          entity_es_enc = tf.contrib.layers.fully_connected(#A:m
                    inputs=entity_enc,
                    num_outputs=hps.hidden_dim,
                    activation_fn=activation_fn,
                    biases_initializer=self.trunc_norm_init,
                    scope="entity_es_enc", reuse=tf.AUTO_REUSE)
          
          sent_es_enc = tf.contrib.layers.fully_connected(#B:n
                inputs=sent_enc,
                num_outputs=hps.hidden_dim,
                activation_fn=activation_fn,
                biases_initializer=self.trunc_norm_init,
                scope="sent_es_enc", reuse=tf.AUTO_REUSE)
          
          new_entity_es_enc = tf.matmul(self._entities2sents_DAD, sent_es_enc)  
          new_sent_es_enc = tf.matmul(self._sents2entities_DAD, entity_es_enc)
          
          entity_self_enc = tf.contrib.layers.fully_connected(#A:m
                    inputs=entity_enc,
                    num_outputs=hps.hidden_dim,
                    activation_fn=activation_fn,
                    biases_initializer=self.trunc_norm_init,
                    scope="entity_self_enc", reuse=tf.AUTO_REUSE)
          
          sent_self_enc = tf.contrib.layers.fully_connected(#B:n
                inputs=sent_enc,
                num_outputs=hps.hidden_dim,
                activation_fn=activation_fn,
                biases_initializer=self.trunc_norm_init,
                scope="sent_self_enc", reuse=tf.AUTO_REUSE)
          
          if hps.use_gcn_entity:
              sent_enc = tf.nn.relu(new_sent_ss_enc + new_sent_es_enc + sent_self_enc)
              entity_enc = tf.nn.relu(new_entity_ee_enc + new_entity_es_enc + entity_self_enc)
          else:
              sent_enc = tf.nn.relu(new_sent_es_enc + sent_self_enc)
              entity_enc = tf.nn.relu(new_entity_es_enc + entity_self_enc)
        
      return sent_enc, entity_enc

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    word_vocab_metadata_path = os.path.join(train_dir, "word_vocab_metadata.tsv")
    self._word_vocab.write_metadata(word_vocab_metadata_path) # write metadata file
    entity_vocab_metadata_path = os.path.join(train_dir, "entity_vocab_metadata.tsv")
    self._entity_vocab.write_metadata(entity_vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_kbsumm(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._word_vocab.size() # size of the vocabulary

    with tf.variable_scope('kbsumm'):
        # Some initializers
        self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

    # Add embedding matrix (shared by the encoder and decoder inputs)
    with tf.variable_scope('kbsumm_word_embedding'): 
        word_embedding=np.fromfile(common.path_word_vocab_embeddings, np.float32).astype(np_float64)#.astype(np.float32)
        word_embedding=np.reshape(word_embedding,[-1,hps.word_emb_dim])
        word_embedding=np.vstack([np.random.normal(0,1,[self._word_vocab.special_size(), hps.word_emb_dim]).astype(np_float64), word_embedding])
        word_embedding=tf.get_variable(initializer=word_embedding, dtype=tf_float64, name='word_embedding')
        
        #embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf_float64, initializer=self.trunc_norm_init)
        #if hps.mode=="train": self._add_emb_vis(word_embedding) # add to tensorboard
        emb_sents_inputs = tf.nn.embedding_lookup(word_embedding, self._sent_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        
        emb_mention_inputs = tf.nn.embedding_lookup(word_embedding, self._mention_batch)

    if hps.use_entity_embedding:
        #Add the entity encoder
        with tf.variable_scope('kbsumm_entity_embedding'): 
            if hps.entity_vocab_size // 10000 == 4:
                path_entity_vocab_embeddings=common.path_entity_vocab_embeddings
            elif hps.entity_vocab_size // 10000 == 10:
                path_entity_vocab_embeddings=common.path_entity_vocab_embeddings_10
            elif hps.entity_vocab_size // 10000 == 20:
                path_entity_vocab_embeddings=common.path_entity_vocab_embeddings_20
            entity_embedding=np.fromfile(path_entity_vocab_embeddings, np.float32).astype(np_float64)#.astype(np.float64)
            entity_embedding=np.reshape(entity_embedding,[-1,hps.entity_emb_dim])
            entity_embedding=np.vstack([np.random.normal(0,1,[self._entity_vocab.special_size(), hps.entity_emb_dim]).astype(np_float64), entity_embedding])
            entity_embedding=tf.get_variable(initializer=entity_embedding, dtype=tf_float64, name='entity_embedding', trainable=True)
    
            #if hps.mode=="train": self._add_emb_vis(entity_embedding) # add to tensorboard
        emb_entity_inputs = tf.nn.embedding_lookup(entity_embedding, self._entity_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        
    else:
        emb_entity_inputs=None
    
    # Entity encodings
    self._entity_enc = self._add_entity_encoder(emb_entity_inputs, emb_mention_inputs) 
    
    # Sent encodings.
    sent_encoding = self._add_sents_encoder(emb_sents_inputs, self._sent_nums, self._sent_lens)
    # Position_encodings
    pos_embedding = None
    if hps.pos_emb_type == "pos_emb":
        max_len = hps.max_position
        dim = hps.hidden_dim
        pe = np.zeros([max_len, dim], dtype=np.float32) 
    
        positions = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        
        div_term = np.exp((np.arange(0, dim, 2) *
                                  -(math.log(10000.0) / dim)))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        pos_embedding = tf.constant(pe)
    self._sent_enc = self._add_sents_pos_emb(sent_encoding, pos_embedding)

    # Add the decoder.
    with tf.variable_scope('gcn'):
      sent_gcn_enc, entity_gcn_enc = self._add_gcn()

    # Add the output projection to obtain the vocabulary distribution
    with tf.variable_scope('output_projection'):
      #w_sent = tf.get_variable('w_sent', [hps.hidden_dim, hps.max_sent_num], dtype=tf_float64, initializer=self.trunc_norm_init)
      #w_sent_t = tf.transpose(w)
      #v_sent = tf.get_variable('v_sent', [hps.max_sent_num], dtype=tf_float64, initializer=self.trunc_norm_init)
      sent_scores = tf.contrib.layers.fully_connected(#B:n
                inputs=sent_gcn_enc,
                num_outputs=1,
                activation_fn=None,#tf.nn.softmax,
                biases_initializer=self.trunc_norm_init,
                scope="sent_scores")
      #self._sent_scores = tf.nn.softmax(tf.squeeze(sent_scores, 2)) * self._pos_mask
      #self._sent_scores = tf.div(self._sent_scores, tf.expand_dims(tf.reduce_sum(self._sent_scores, 1),1))
      sent_scores = tf.squeeze(sent_scores, 2)  * self._pos_mask + tf.float32.min * (1 - self._pos_mask)
      self._sent_scores = tf.nn.softmax(sent_scores) 
      sent_scores = tf.concat([sent_scores, tf.fill([hps.batch_size, hps.max_sent_num - self._max_sent_num], tf.float32.min)], 1)
      #sent_gcn_enc = tf.concat([sent_gcn_enc, tf.zeros([hps.batch_size, hps.max_sent_num - self._max_sent_num])], 1)
      #sent_scores = tf.nn.softmax(tf.nn.xw_plus_b(sent_gcn_enc, w_sent, v_sent)) * self._sent_mask

      #w_entity = tf.get_variable('w_entity', [hps.hidden_dim, self._max_entity_num], dtype=tf_float64, initializer=self.trunc_norm_init)
      #w_entity_t = tf.transpose(w)
      #v_entity = tf.get_variable('v_entity', [hps.max_entity_num], dtype=tf_float64, initializer=self.trunc_norm_init)
      #entity_scores = tf.nn.softmax(tf.nn.xw_plus_b(entity_gcn_enc, w_entity, v_entity)) * self._entity_mask
      entity_scores = tf.contrib.layers.fully_connected(#B:n
                    inputs=entity_gcn_enc,
                    num_outputs=1,
                    activation_fn=None,#tf.nn.softmax,
                    biases_initializer=self.trunc_norm_init,
                    scope="entity_scores")
      #self._entity_scores = tf.nn.softmax(tf.squeeze(entity_scores, 2))   * self._entity_mask
      #self._entity_scores = tf.div(self._entity_scores, tf.expand_dims(tf.reduce_sum(self._entity_scores, 1), 1))
      entity_scores = tf.squeeze(entity_scores, 2)  * self._entity_mask + tf.float32.min * (1 - self._entity_mask)
      self._entity_scores = tf.nn.softmax(entity_scores)
      entity_scores = tf.concat([entity_scores, tf.fill([hps.batch_size, hps.max_entity_num - self._max_entity_num], tf.float32.min)], 1)

      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          sent_label_batch = tf.concat([self._sent_label_batch, tf.zeros([hps.batch_size, hps.max_sent_num - self._max_sent_num], dtype=tf_float64)], 1)
          self._sent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=sent_label_batch, logits=sent_scores) 
          #self._sent_loss = sent_label_batch * tf.log(sent_scores)
          
          entity_label_batch = tf.concat([self._entity_label_batch, tf.zeros([hps.batch_size, hps.max_entity_num - self._max_entity_num], dtype=tf_float64)], 1)
          self._entity_loss = tf.nn.softmax_cross_entropy_with_logits(labels=entity_label_batch, logits=entity_scores) 
          #self._entity_loss = entity_label_batch * tf.log(entity_scores)
          
          self._loss = (1 - hps.entity_lambda) * self._sent_loss + hps.entity_lambda * self._entity_loss
          #self._loss = self._entity_loss 
          
          if hps.lambda_ee_train>0 and hps.use_gcn_entity:
              ee_loss=self._add_entities2entities_loss(emb_entity_inputs, self._entities2entities)
              self._loss += hps.lambda_ee_train*ee_loss
          
          if hps.rl_lambda > 0:
              rl_sent_label_batch = tf.concat([self._rl_sent_label_batch, tf.zeros([hps.batch_size, hps.max_sent_num - self._max_sent_num], dtype=tf_float64)], 1)
              self._rl_sent_loss = tf.nn.softmax_cross_entropy_with_logits(labels=rl_sent_label_batch, logits=sent_scores) 
              #self._rl_sent_loss = rl_sent_label_batch * tf.log(entity_scores)
              
              if hps.use_gcn_entity:
                  rl_entity_label_batch = tf.concat([self._rl_entity_label_batch, tf.zeros([hps.batch_size, hps.max_entity_num - self._max_entity_num], dtype=tf_float64)], 1)
                  self._rl_entity_loss = tf.nn.softmax_cross_entropy_with_logits(labels=rl_entity_label_batch, logits=entity_scores) 
              #self._rl_entity_loss = rl_entity_label_batch * tf.log(entity_scores)
              
              self._sampled_sents = tf.multinomial(self._sent_scores, hps.sample_sent_num * 10)
              self._sampled_entities = tf.multinomial(self._entity_scores, hps.sample_entity_num * 10)
              
              #self._greedy_sents = tf.multinomial(self._sent_scores, hps.sample_sent_num * 10)
              #self._greedy_entities = tf.multinomial(self._entity_scores, hps.sample_entity_num * 10)
              self._loss = (1 - hps.rl_lambda) *self._loss + hps.rl_lambda * (self._rl_reward_batch * ((1 - hps.entity_lambda) * self._rl_sent_loss + hps.entity_lambda * self._rl_entity_loss))
          
          self._loss = tf.reduce_mean(self._loss)
          tf.summary.scalar('loss', self._loss)

    #if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
    #  assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
    #  final_dists = final_dists[0]
    #  topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
    # self._topk_log_probs = tf.log(topk_probs)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._loss
    tvars = tf.trainable_variables()
    print(tvars)
    for item in self.frozen_vars:
        tvars.remove(item)
    print(tvars)
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def sampler(self, sess, batch):
      feed_dict = self._make_feed_dict(batch, use_rl=False)
      to_return = {
        'sampled_sents': self._sampled_sents,
        'sampled_entities': self._sampled_entities, 
        "sent_scores": self._sent_scores,
        "entity_scores": self._entity_scores
      }
      return sess.run(to_return, feed_dict)

  def greedy_decode(self, sess, batch):

    feed_dict = self._make_feed_dict(batch, use_rl=False)

    to_return = {
      "sent_scores": self._sent_scores,
      "entity_scores": self._entity_scores,
      "data_name": self._data_name_batch
    }

    results = sess.run(to_return, feed_dict=feed_dict) 

    return results["sent_scores"], results["entity_scores"], results["data_name"]

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_kbsumm()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
        #'scores':self._sent_scores2,
        #'labels':self._sent_label_batch,
        "data_name": self._data_name_batch
    }
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):

    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run(
        [self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = dec_in_state[0]
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, img_states, dec_init_states, prev_coverage):

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    #cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state, axis=0) for state in dec_init_states]
    #new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = new_h

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self._data_name_batch:self._data_name_batch
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self._attn_dists,
      "data_name": self._data_name_batch
    }

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [results['states'][i, :] for i in xrange(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.exp(tf.float32.min)+tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


















