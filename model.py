

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

from tensorflow.python.ops import nn_ops
import common
tf_float64=tf.float32
np_float64=np.float32
FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, word_vocab, entity_vocab):
    self._hps = hps
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    self._max_entity_num = tf.placeholder(tf.int32, name='max_entity_num')
    self._max_mention_len = tf.placeholder(tf.int32, name='max_mention_len')
    
    # encoder part
    self._data_name_batch = tf.placeholder(tf.int32, [hps.batch_size,20], name='data_name_batch')
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf_float64, [hps.batch_size, None], name='enc_padding_mask')
    
    self._entity_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='entity_batch') 
    self._entity_nums = tf.placeholder(tf.int32, [hps.batch_size], name='entity_nums')
    self._mention_lens = tf.placeholder(tf.int32, [hps.batch_size, None], name='mention_lens')
    self._mention_batch = tf.placeholder(tf.int32, [hps.batch_size, None, None], name='mention_batch')
    self._mention_mask = tf.placeholder(tf_float64, [hps.batch_size, None, None], name='mention_mask')
    
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf_float64, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf_float64, [hps.batch_size, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._data_name_batch] = batch.data_name_batch
    
    feed_dict[self._max_entity_num] = batch.max_entity_num
    feed_dict[self._max_mention_len] = batch.max_mention_len
    
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    
    feed_dict[self._entity_batch] = batch.entity_batch
    feed_dict[self._entity_nums] = batch.entity_nums
    feed_dict[self._mention_lens] = batch.mention_lens
    feed_dict[self._mention_batch] = batch.mention_batch
    feed_dict[self._mention_mask] = batch.mention_mask
    
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
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

  def _add_text_encoder(self, encoder_inputs):

    with tf.variable_scope('seq2seq_text_encoder'): 
      cell_fw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2) #tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = self._get_rnn_cell("GRU", hidden_dim=self._hps.hidden_dim/2) #tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf_float64, 
                                                                          sequence_length=self._enc_lens, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    
    return encoder_outputs, fw_st, bw_st 

  def _add_entity_encoder(self, entity_inputs, mention_inputs):
    hps = self._hps
    
    with tf.variable_scope('seq2seq_mention_encoder'): 
        cell_fw_me = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim/2) 
        cell_bw_me = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim/2) 
          
        mention_inputs = tf.reshape(mention_inputs, [-1, self._max_mention_len, hps.word_emb_dim])
        mention_lens = tf.reshape(self._mention_lens, [-1])
          
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_me, cell_bw_me, mention_inputs, dtype=tf_float64, 
                                                                              sequence_length=mention_lens, swap_memory=True)
        #self._mention_outputs = encoder_outputs
        #self._mention_outputs = tf.concat(axis=2, values=encoder_outputs) 
          
        #encoder_outputs = tf.reshape(encoder_outputs, [hps.batch_size, self._max_entity_num, self._max_mention_len, hps.hidden_dim])
        
        mention_encs = self._reduce_states(fw_st, bw_st, 1, "mention")
        mention_encs = tf.reshape(mention_encs, [hps.batch_size, self._max_entity_num, hps.hidden_dim]) 
        
        if hps.use_entity_embedding:
            mention_encs = tf.concat([mention_encs, entity_inputs], 2)
            mention_encs = tf.contrib.layers.fully_connected(
                inputs=mention_encs,
                num_outputs=hps.hidden_dim,
                activation_fn=tf.tanh,
                biases_initializer=self.trunc_norm_init,
                scope="mention_encs")
        
        entity_enc = tf.reduce_mean(mention_encs, 1) 
        #entity_enc = tf.zeros_like(entity_enc)
    ''' 
    with tf.variable_scope('seq2seq_entity_encoder'): 
        cell_fw_en = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim/2) 
        cell_bw_en = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim/2) 
        
        entity_nums = self._entity_nums
          
        (_, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_en, cell_bw_en, mention_encs, dtype=tf_float64, 
                                                                              sequence_length=entity_nums, swap_memory=True)
        #encoder_outputs = tf.concat(axis=2, values=encoder_outputs) 
          
        #encoder_outputs = tf.reshape(encoder_outputs, [hps.batch_size, self._max_entity_num, self._max_mention_len, hps.hidden_dim])
          
        entity_encs = self._reduce_states(fw_st, bw_st, 1, "entity")
        entity_encs = tf.reshape(entity_encs, [hps.batch_size, self._max_entity_num, hps.hidden_dim]) 
    '''    
    return mention_encs, entity_enc

  def _reduce_states(self, fw_st, bw_st, concat_axis=1, scope="text"):
    
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('seq2seq_reduce_final_st_'+scope):

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

  def _add_decoder(self, inputs):

    hps = self._hps
    cell = self._get_rnn_cell("GRU", hidden_dim=hps.hidden_dim) #tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    
    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, self._entity_enc, 
                                                                         cell, initial_state_attention=(hps.mode=="decode"), 
                                                                          pointer_gen=hps.pointer_gen, 
                                                                         use_coverage=hps.coverage, prev_coverage=prev_coverage)

    return outputs, out_state, attn_dists, p_gens, coverage

  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('seq2seq_final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._word_vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs), dtype=tf_float64)
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]
      #print(len(final_dists))
      #print(final_dists))
      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_vizfinal_dists
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

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._word_vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
        # Some initializers
        self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

    # Add embedding matrix (shared by the encoder and decoder inputs)
    with tf.variable_scope('seq2seq_word_embedding'): 
        word_embedding=np.fromfile(common.path_word_vocab_embeddings, np_float64).astype(np_float64)#.astype(np.float32)
        word_embedding=np.reshape(word_embedding,[-1,hps.word_emb_dim])
        word_embedding=np.vstack([np.zeros([self._word_vocab.special_size(), hps.word_emb_dim], np_float64), word_embedding])
        word_embedding=tf.get_variable(initializer=word_embedding, dtype=tf_float64, name='word_embedding')
        
        #embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf_float64, initializer=self.trunc_norm_init)
        #if hps.mode=="train": self._add_emb_vis(word_embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(word_embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(word_embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        #self.emb_enc_inputs = emb_enc_inputs
        
        enb_mention_inputs = tf.nn.embedding_lookup(word_embedding, self._mention_batch)

    # Add the text encoder.
    enc_outputs, enc_fw_st, enc_bw_st = self._add_text_encoder(emb_enc_inputs)
    self._enc_states = enc_outputs
    
    if hps.use_entity_embedding:
        #Add the entity encoder
        with tf.variable_scope('seq2seq_entity_embedding'): 
            if hps.entity_vocab_size // 10000 == 5:
                path_entity_vocab_embeddings=common.path_entity_vocab_embeddings_5
            elif hps.entity_vocab_size // 10000 == 10:
                path_entity_vocab_embeddings=common.path_entity_vocab_embeddings_10
            elif hps.entity_vocab_size // 10000 == 20:
                path_entity_vocab_embeddings=common.path_entity_vocab_embeddings_20
            entity_embedding=np.fromfile(path_entity_vocab_embeddings, np_float64).astype(np_float64)#.astype(np.float64)
            entity_embedding=np.reshape(entity_embedding,[-1,hps.entity_emb_dim])
            entity_embedding=np.vstack([np.zeros([self._entity_vocab.special_size(), hps.entity_emb_dim], np_float64), entity_embedding])
            entity_embedding=tf.get_variable(initializer=entity_embedding, dtype=tf_float64, name='entity_embedding')
    
            #if hps.mode=="train": self._add_emb_vis(entity_embedding) # add to tensorboard
        emb_entity_inputs = tf.nn.embedding_lookup(entity_embedding, self._entity_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
    else:
        emb_entity_inputs=None
    self._entity_states, self._entity_enc = self._add_entity_encoder(emb_entity_inputs, enb_mention_inputs) #entity_inputs, mention_inputs, occur_inputs, seq_len

    # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
    self._dec_in_state = self._reduce_states(enc_fw_st, enc_bw_st, 1, "text")

    # Add the decoder.
    with tf.variable_scope('seq2seq_decoder'):
      decoder_outputs, self._dec_out_state, self._attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)

    # Add the output projection to obtain the vocabulary distribution
    with tf.variable_scope('seq2seq_output_projection'):
      w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf_float64, initializer=self.trunc_norm_init)
      w_t = tf.transpose(w)
      v = tf.get_variable('v', [vsize], dtype=tf_float64, initializer=self.trunc_norm_init)
      vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
      for i,output in enumerate(decoder_outputs):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

      vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self._attn_dists)
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists



      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('seq2seq_loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs)
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

          else: # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('seq2seq_coverage_loss'):
              self._coverage_loss = _coverage_loss(self._attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)

    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      #print(len(final_dists))
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      if hps.dec_method == "beam-search":
          topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
          self._topk_log_probs = tf.log(topk_probs)
      elif hps.dec_method == "greedy":
          self._top_ids = tf.argmax(final_dists, axis=1)


  def _add_train_op(self, gpu=0):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:%d" % gpu):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:%d" % gpu):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self, gpu=0):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:%d" % gpu):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)
    
    tvars = []
    tvars.extend(tf.trainable_variables())
    return tvars

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
        "data_name": self._data_name_batch#, 
        #"enc_states": self._enc_states, 
        #"entity_enc": self._entity_enc, 
        #"emb_enc_inputs": self.emb_enc_inputs, 
        #"enc_lens": self._enc_lens, 
        #"mention_outputs": self._mention_outputs
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
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
    (enc_states, entity_states, entity_enc, dec_in_state, global_step) = sess.run(
        [self._enc_states, self._entity_states, self._entity_enc, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = dec_in_state[0]
    return enc_states, entity_states, entity_enc, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, entity_states, entity_enc, dec_init_states, prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    #cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state, axis=0) for state in dec_init_states]
    #new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = new_h

    feed = {
        self._enc_states: enc_states,
        self._entity_states: entity_states,
        self._entity_enc: entity_enc,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens]))
    }

    to_return = {
      "ids": self._topk_ids if self._hps.dec_method == "beam-search" else self._top_ids,
      #"probs": self._topk_log_probs if self._hps.dec_method == "beam-search" else None,
      "states": self._dec_out_state,
      "attn_dists": self._attn_dists
    }
    
    if self._hps.dec_method == "beam-search":
        to_return["probs"] = self._topk_log_probs

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

    return results['ids'], results['probs'] if self._hps.dec_method == "beam-search" else None, new_states, attn_dists, p_gens, new_coverage


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


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss




















