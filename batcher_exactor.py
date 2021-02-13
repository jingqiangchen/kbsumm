

from queue import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data
import copy

from batcher_generator import Batch as GenBatch

np_float64=np.float32
MIN_INV = 1e-10

def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target

class Example(object):

  def __init__(self, data_name, text, summary, summary2, sent_relas, entities2sents, entities2entities, summary_labels, entities, 
         entity_labels, entities2mentions, word_vocab, entity_vocab, hps):

    self.hps = hps
    
    #stop_decoding = data.STOP_DECODING
    #estop_decoding = data.ESTOP_DECODING
    start_decoding = word_vocab.word2id(data.START_DECODING)
    stop_decoding = word_vocab.word2id(data.STOP_DECODING)
    mention_seg = word_vocab.word2id(data.MENTION_SEG)
    
    self.data_name = data_name
    #print(data_name)
    # Process the article
    
    self.summary = summary
    summary_words = summary.split() # list of strings
    summary_ids = [word_vocab.word2id(w) for w in summary_words] # list of word ids; OOVs are represented by the id for UNK token
    self.dec_input, self.target = get_dec_inp_targ_seqs(summary_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)
    
    self.summary2 = summary2
    #print(summary2)
    
    sents = text.split("\n")
    self.sents = sents
    sents = [sent.split()[:hps.max_enc_steps] if len(sent.split()) > hps.max_enc_steps else  sent.split() for sent in sents]
    #sents.append([stop_decoding])

    self.sent_num = len(sents)
    self.sent_lens = [len(sent) for sent in sents] 
    #print("self.sent_lens",self.sent_lens)
    self.sent_inputs = []
    for sent in sents:
        self.sent_inputs.append([word_vocab.word2id(w) for w in sent])
    
    entities = entities.split("|||")
    #entities.append(estop_decoding)
    self.entity_num = len(entities)
    self.entity_ids = [entity_vocab.word2id(w) for w in entities]
    
    self.mention_lens = []
    self.mention_ids = []
    self.mentions = []
    mentions = entities2mentions.split("|||")
    mentions = [mention.split("||") for mention in mentions]
    self.max_mention_len=0
    for mention_line in mentions:
        m2=[]
        for mention in mention_line:
            words=mention.split()
            m2.extend([word_vocab.word2id(w) for w in words])
            m2.append(mention_seg)
        del m2[-1]
        self.mention_ids.append(m2)
        self.mentions.append(mention_line)
        self.mention_lens.append(len(m2))
        if self.max_mention_len<len(m2):self.max_mention_len=len(m2)
    
    self.sent_relas = np.reshape(np.frombuffer(sent_relas, np.int32), [self.sent_num, self.sent_num]).astype(np_float64)
    #self.sent_relas = np.zeros([self.sent_num,self.sent_num],dtype=np_float64)
    #self.sent_relas[:self.sent_num-1,:self.sent_num-1]=sent_relas[:,:]
    D_left=np.linalg.pinv(np.diag(np.sum(self.sent_relas, 1) + MIN_INV) ** 0.5)
    D_right = np.linalg.pinv(np.diag(np.sum(self.sent_relas, 0) + MIN_INV) ** 0.5)
    self.sent_relas_DAD = np.matmul(np.matmul(D_left, self.sent_relas), D_right)
    
    self.entities2sents = np.reshape(np.frombuffer(entities2sents, np.int32), [self.entity_num, self.sent_num]).astype(np_float64)
    #self.entities2sents = np.zeros([self.entity_num, self.sent_num], dtype=np_float64)
    #self.entities2sents[:self.entity_num-1,:self.sent_num-1]=entities2sents[:,:]
    D_left=np.linalg.pinv(np.diag(np.sum(self.entities2sents, 1) + MIN_INV) ** 0.5)
    D_right = np.linalg.pinv(np.diag(np.sum(self.entities2sents, 0) + MIN_INV) ** 0.5)
    self.entities2sents_DAD = np.matmul(np.matmul(D_left, self.entities2sents), D_right)
    self.sents2entities_DAD = np.matmul(np.matmul(D_right, self.entities2sents.transpose(1, 0)), D_left)
    
    self.entities2entities = np.reshape(np.frombuffer(entities2entities, np.int32), [self.entity_num, self.entity_num]).astype(np_float64)
    self.entities2entities = self.entities2entities * (1-np.eye(self.entity_num))
    #self.entities2entities = np.zeros([self.entity_num, self.entity_num], dtype=np_float64)
    #self.entities2entities[:self.entity_num-1,:self.entity_num-1]=entities2entities[:,:]
    D_left=np.linalg.pinv(np.diag(np.sum(self.entities2entities, 1) + MIN_INV) ** 0.5)
    D_right = np.linalg.pinv(np.diag(np.sum(self.entities2entities, 0) + MIN_INV) ** 0.5)
    self.entities2entities_DAD = np.matmul(np.matmul(D_left, self.entities2entities), D_right)
    
    sent_label = np.frombuffer(summary_labels, np_float64)
    self.sent_label = np.zeros([self.sent_num], np_float64)
    self.sent_label[:self.sent_num]=sent_label[:]
    #self.sent_label[-1]=0.5
    #print("sent_label",self.sent_label)
    
    entity_label = np.frombuffer(entity_labels, np.int32).astype(np_float64)
    self.entity_label = np.zeros([self.entity_num], np_float64)
    self.entity_label[:self.entity_num]=entity_label[:]
    #self.entity_label[-1]=0.5
    #print("entity_label",self.entity_label)

    # Store the original strings
    self.original_text = text
    self.original_summary_labels = summary_labels
    self.original_entity_labels = entity_labels


  def pad_sent_inputs(self, max_sent_num, max_sent_len, pad_id):
    
    for sent_input in self.sent_inputs:
        while len(sent_input) < max_sent_len:
            sent_input.append(pad_id)
    
    while len(self.sent_inputs) < max_sent_num:
        self.sent_inputs.append([pad_id] * max_sent_len)
        
  def pad_entity_ids(self, max_entity_num, pad_id):
    while len(self.entity_ids) < max_entity_num:
      self.entity_ids.append(pad_id)
      
  def pad_decoder_inp_targ(self, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, word_vocab):
    self.word_vocab = word_vocab
    self.hps = hps
    if example_list is None: return
    self.pad_id = word_vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_data(example_list, hps) # initialize the input to the encoder
    self.store_orig_strings(example_list) # store the original strings
    self.word_vocab = word_vocab

  def init_data(self, example_list, hps):
    #data_name, text, sent_relas, entities2sents, entities2entities, summary_labels, entities, entity_labels, entities2mentions, entities2mentions_count
    # Determine the maximum length of the encoder input sequence in this batch
    
    max_sent_num = max([ex.sent_num for ex in example_list])
    max_sent_len = max([max(ex.sent_lens) for ex in example_list])
    max_entity_num = max([ex.entity_num for ex in example_list])
    max_mention_len = max([ex.max_mention_len for ex in example_list])
    #print("max_sent_len",max_sent_len)
    self.max_sent_num = max_sent_num
    self.max_sent_len = max_sent_len
    
    self.max_entity_num = max_entity_num
    self.max_mention_len = max_mention_len

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_sent_inputs(max_sent_num, max_sent_len, self.pad_id)
      ex.pad_entity_ids(max_entity_num, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.data_name_batch = np.zeros((hps.batch_size, 20), dtype=np.int32)
    self.sent_batch = np.zeros((hps.batch_size, max_sent_num, max_sent_len), dtype=np.int32)
    
    self.sent_str_batch = []
    self.summary_str_batch = []
    self.summary_str_batch2 = []
    self.mention_str_batch = []
    
    self.entity_nums = np.zeros((hps.batch_size), dtype=np.int32)
    self.entity_batch = np.zeros((hps.batch_size, max_entity_num), dtype=np.int32)
    self.mention_lens = np.zeros((hps.batch_size, max_entity_num), dtype=np.int32)
    self.mention_batch = np.zeros((hps.batch_size, max_entity_num, max_mention_len), dtype=np.int32)
    self.mention_batch.fill(self.pad_id)
    self.mention_mask = np.zeros((hps.batch_size, max_entity_num, max_mention_len), dtype=np_float64)
    
    self.sent_nums = np.zeros((hps.batch_size), dtype=np.int32)
    self.sent_lens = np.zeros((hps.batch_size, max_sent_num), dtype=np.int32)
    self.sent_padding_mask = np.zeros((hps.batch_size, max_sent_num, max_sent_len), dtype=np_float64)
    
    self.sent_relas = np.zeros((hps.batch_size, max_sent_num, max_sent_num), dtype=np_float64)
    self.sent_relas_DAD = np.zeros((hps.batch_size, max_sent_num, max_sent_num), dtype=np_float64)
    
    self.entities2sents = np.zeros((hps.batch_size, max_entity_num, max_sent_num), dtype=np_float64)
    self.entities2sents_DAD = np.zeros((hps.batch_size, max_entity_num, max_sent_num), dtype=np_float64)
    self.sents2entities_DAD = np.zeros((hps.batch_size, max_sent_num, max_entity_num), dtype=np_float64)
    
    self.entities2entities = np.zeros((hps.batch_size, max_entity_num, max_entity_num), dtype=np_float64)
    self.entities2entities_DAD = np.zeros((hps.batch_size, max_entity_num, max_entity_num), dtype=np_float64)
    
    self.pos_batch = np.zeros((hps.batch_size, max_sent_num), dtype=np.int32)
    self.pos_mask = np.zeros((hps.batch_size, max_sent_num), dtype=np_float64)
    self.entity_mask = np.zeros((hps.batch_size, max_entity_num), dtype=np_float64)
    self.entities2entities_mask = np.zeros((hps.batch_size, max_entity_num), dtype=np_float64)
    
    self.sent_label_batch = np.zeros((hps.batch_size, max_sent_num), dtype=np_float64)
    self.entity_label_batch = np.zeros((hps.batch_size, max_entity_num), dtype=np_float64)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.data_name_batch[i,:]=[ord(x) for x in ex.data_name[:20]]
      
      self.sent_batch[i, :, :] = ex.sent_inputs[:]
      
      self.sent_str_batch.append(ex.sents)
      self.summary_str_batch.append(ex.summary)
      self.summary_str_batch2.append(ex.summary2)
      self.mention_str_batch.append(ex.mentions)
      
      self.sent_nums[i] = ex.sent_num
      self.sent_lens[i, :ex.sent_num] = ex.sent_lens[:]
      for j in xrange(ex.sent_num):
          for k in xrange(ex.sent_lens[j]):
              self.sent_padding_mask[i][j][k] = 1
              
      self.entity_batch[i, :] = ex.entity_ids[:]
      
      self.mention_lens[i, :ex.entity_num] = ex.mention_lens[:]
      
      try:
          for I in xrange(len(ex.mention_ids)):
              for j in xrange(len(ex.mention_ids[I])):
                self.mention_batch[i,I,j]=ex.mention_ids[I][j]
      except:
          print(ex.data_name)
          
      #try:
      for I in xrange(len(ex.mention_lens)):
        for j in xrange(ex.mention_lens[I]):
            self.mention_mask[i,I,j]=1.0
      #except:
     #     print("mention_lens:"+ex.data_name)
          
      self.entity_nums[i] = ex.entity_num
      
      for j in xrange(ex.sent_num):
          for k in xrange(ex.sent_num):
              self.sent_relas[i, j, k] = ex.sent_relas[j][k]
              self.sent_relas_DAD[i, j, k] = ex.sent_relas_DAD[j][k]
      
      for j in xrange(ex.entity_num):
          for k in xrange(ex.sent_num):
              self.entities2sents[i, j, k] = ex.entities2sents[j][k]
              self.entities2sents_DAD[i, j, k] = ex.entities2sents_DAD[j][k]
              self.sents2entities_DAD[i, k, j] = ex.sents2entities_DAD[k][j]
      
      for j in xrange(ex.entity_num):
          for k in xrange(ex.entity_num):
              self.entities2entities[i, j, k] = ex.entities2entities[j][k]
              self.entities2entities_DAD[i, j, k] = ex.entities2entities_DAD[j][k]
              
      self.pos_batch[i, :ex.sent_num] = np.arange(0, ex.sent_num, dtype=np.int32)
      self.pos_mask[i, :ex.sent_num] = np.ones(ex.sent_num)
      self.entity_mask[i, :ex.entity_num] = np.ones(ex.entity_num)
      self.entities2entities_mask[i, :ex.entity_num] = np.ones(ex.entity_num)
      
      sum_sl = np.sum(ex.sent_label)
      if sum_sl == 0:
          self.sent_label_batch[i, :ex.sent_num] = ex.sent_label[:]
      else:
          self.sent_label_batch[i, :ex.sent_num] = ex.sent_label[:]/sum_sl
      
      sum_el = np.sum(ex.entity_label)    
      if sum_el == 0:
          self.entity_label_batch[i, :ex.entity_num] = ex.entity_label[:]
      else:
          self.entity_label_batch[i, :ex.entity_num] = ex.entity_label[:]/sum_el
       
        
    # Decode, Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
    self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np_float64)
    
    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      for j in xrange(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""
    self.original_text = [ex.original_text for ex in example_list] # list of lists
    self.original_summary_labels = [ex.original_summary_labels for ex in example_list] # list of lists
    self.original_entity_labels = [ex.original_entity_labels for ex in example_list] # list of list of lists

  def tile_for_greedy(self, summary_exts, max_enc_steps, all_entity_indices, max_enc_len=None):
    batch=GenBatch(None,None,None)
    batch_size = len(summary_exts)
    
    batch.max_entity_num = self.max_entity_num
    batch.max_mention_len = self.max_mention_len
    
    batch.data_name_batch = np.tile(self.data_name_batch, 1)
    batch.dec_batch = np.tile(self.dec_batch, 1)
    batch.target_batch = np.tile(self.target_batch, 1)
    batch.dec_padding_mask = np.tile(self.dec_padding_mask, 1)
    
    mention_batch_shape = np.shape(self.mention_batch)
    batch.entity_nums = np.zeros([batch_size], dtype=np.int32)
    batch.entity_batch = np.zeros([batch_size, self.max_entity_num], dtype=np.int32)
    batch.mention_batch = np.zeros([batch_size, self.max_entity_num, mention_batch_shape[2]], dtype=np.int32)
    batch.mention_mask = np.zeros([batch_size, self.max_entity_num, mention_batch_shape[2]], dtype=np_float64)
    
    batch.mention_lens = np.zeros((batch_size, self.max_entity_num), dtype=np.int32)
    
    all_summary_ext_words=[]
    for index in range(batch_size):
        summary_ext = summary_exts[index]
        summary_ext_words = summary_ext.split()
        if len(summary_ext_words) > max_enc_steps:
          summary_ext_words = summary_ext_words[:max_enc_steps]
        all_summary_ext_words.append(summary_ext_words)
    
    if max_enc_len is None:
        max_enc_len = max([len(summary_ext) for summary_ext in all_summary_ext_words])
    
    batch.enc_batch = np.zeros([batch_size, max_enc_len], dtype=np.int32)
    batch.enc_lens = np.zeros([batch_size], dtype=np.int32)
    batch.enc_padding_mask = np.zeros([batch_size, max_enc_len], dtype=np_float64)
    
    if self.hps.pointer_gen:
        all_article_oovs=[]
        all_enc_input_extend_vocab=[]
        all_extend_vocab=[]
        for index in range(batch_size):
            summary_ext_words=all_summary_ext_words[index]
        
            enc_input_extend_vocab, article_oovs = data.article2ids(summary_ext_words, self.word_vocab)
            extend_vocab = data.abstract2ids(summary_ext_words, self.word_vocab, article_oovs)
            all_article_oovs.append(article_oovs)
            all_enc_input_extend_vocab.append(enc_input_extend_vocab)
            all_extend_vocab.append(extend_vocab)
        
        max_art_oovs = max([len(article_oovs) for article_oovs in all_article_oovs])

        batch.max_art_oovs = max_art_oovs
        batch.art_oovs = all_article_oovs
        batch.enc_batch_extend_vocab = np.zeros([batch_size, max_enc_len], dtype=np.int32)
    
    for index in range(batch_size):
        summary_ext_words=all_summary_ext_words[index]
          
        enc_len = len(summary_ext_words) # store the length after truncation but before padding
        enc_input = [self.word_vocab.word2id(w) for w in summary_ext_words] # list of word ids; OOVs are represented by the id for UNK token
        batch.enc_batch[index,:enc_len] = enc_input[:]
        batch.enc_lens[index] = enc_len
        batch.enc_padding_mask[index,:enc_len] = np.ones([enc_len], dtype=np_float64)
        
        batch.mention_lens[index,:]=self.mention_lens[index,:]
        
        entity_indices = all_entity_indices[index]
        entity_num = len(entity_indices)
        
        batch.entity_nums[index] = entity_num
        for i in range(entity_num):
            batch.entity_batch[index,i] = self.entity_batch[index, entity_indices[i]]
            batch.mention_batch[index,i,:] = self.mention_batch[index, entity_indices[i],:]
            batch.mention_mask[i,:,:] = self.mention_mask[index, entity_indices[i],:]

        if self.hps.pointer_gen:
          batch.enc_batch_extend_vocab[index,:enc_len] = all_enc_input_extend_vocab[index][:]
      
    return batch

  def tile_for_gen(self, index, reps, summary_ext, max_enc_steps, entity_indices):
    batch=GenBatch(None,None,None)
    
    batch.data_name_batch=np.tile(self.data_name_batch[index],[reps,1]) #batch.data_name_batch = np.zeros((hps.batch_size, 20), dtype=np.int32)
    
    #batch.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
    #batch.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    #batch.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np_float64)
    summary_ext_words = summary_ext.split()
    if len(summary_ext_words) > max_enc_steps:
      summary_ext_words = summary_ext_words[:max_enc_steps]
    enc_len = len(summary_ext_words) # store the length after truncation but before padding
    
    enc_input = [self.word_vocab.word2id(w) for w in summary_ext_words] # list of word ids; OOVs are represented by the id for UNK token
    batch.enc_batch = np.tile(enc_input, [reps, 1])
    batch.enc_lens = np.tile(enc_len, [reps])
    batch.enc_padding_mask = np.ones([reps, enc_len], dtype=np_float64)
    
    #batch.entity_batch = np.zeros((hps.batch_size, max_entity_num), dtype=np.int32)
    #batch.mention_batch = np.zeros((hps.batch_size, max_entity_num, max_mention_num, max_mention_len), dtype=np.int32)
    #batch.mention_batch.fill(self.pad_id)
    #batch.mention_occur_batch = np.zeros((hps.batch_size, max_entity_num, max_mention_num), dtype=np.int32)
    #batch.entity_nums = np.zeros((hps.batch_size), dtype=np.int32)
    #batch.mention_mask = np.zeros((hps.batch_size, max_entity_num, max_mention_num, max_mention_len), dtype=np_float64)
    entity_num = len(entity_indices)
    batch.max_entity_num = entity_num
    batch.max_mention_len = self.max_mention_len
    entity_batch = np.zeros((entity_num), dtype=np.int32)
    #mention_batch_shape = np.shape(self.mention_batch)
    mention_batch = np.zeros((entity_num, batch.max_mention_len), dtype=np.int32)
    mention_mask = np.zeros([entity_num, batch.max_mention_len], dtype=np_float64)
    mention_lens = np.zeros((entity_num), dtype=np.int32)
    original_entity_batch = self.entity_batch[index]
    original_mention_batch = self.mention_batch[index]
    original_mention_mask = self.mention_mask[index]
    original_mention_lens = self.mention_lens[index]
    for i in range(entity_num):
        entity_batch[i] = original_entity_batch[entity_indices[i]]
        mention_batch[i,:] = original_mention_batch[entity_indices[i],:]
        mention_mask[i,:] = original_mention_mask[entity_indices[i],:]
        mention_lens[i] = original_mention_lens[entity_indices[i]]
    batch.entity_nums = np.tile(entity_num, [reps])
    batch.entity_batch = np.tile(entity_batch, [reps,1])
    batch.mention_batch = np.tile(mention_batch, [reps, 1, 1])
    batch.mention_mask = np.tile(mention_mask, [reps, 1, 1])
    batch.mention_lens = np.tile(mention_lens, [reps, 1])

    #self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    #self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
    #self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np_float64)
    batch.dec_batch = np.tile(self.dec_batch[index], [reps, 1])
    batch.target_batch = np.tile(self.target_batch[index], [reps, 1])
    batch.dec_padding_mask = np.tile(self.dec_padding_mask[index], [reps, 1])
    
    if self.hps.pointer_gen:
      start_decoding = self.word_vocab.word2id(data.START_DECODING)
      stop_decoding = self.word_vocab.word2id(data.STOP_DECODING)
    
      enc_input_extend_vocab, article_oovs = data.article2ids(summary_ext_words, self.word_vocab)
      extend_vocab = data.abstract2ids(summary_ext_words, self.word_vocab, article_oovs)
      _, target = get_dec_inp_targ_seqs(extend_vocab, self.hps.max_dec_steps, start_decoding, stop_decoding)
      
      # Determine the max number of in-article OOVs in this batch
      max_art_oovs = len(article_oovs)
      # Store the in-article OOVs themselves
      art_oovs = article_oovs
      # Store the version of the enc_batch that uses the article OOV ids
      enc_batch_extend_vocab = np.zeros((enc_len), dtype=np.int32)
      enc_batch_extend_vocab[:] = enc_input_extend_vocab[:]
      
      batch.enc_batch_extend_vocab = np.tile(enc_batch_extend_vocab, [reps,1])
      batch.max_art_oovs = max_art_oovs
      batch.art_oovs = np.tile(art_oovs, [reps, 1])
      
      #feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      #feed_dict[self._max_art_oovs] = batch.max_art_oovs

    return batch


class Batcher(object):
  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, word_vocab, entity_vocab, hps, single_pass):
    
    self._data_path = data_path
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab
    self._hps = hps
    self._single_pass = single_pass

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 16 # num threads to fill example queue
      self._num_batch_q_threads = 4  # num threads to fill batch queue
      self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in xrange(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in xrange(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""

    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (data_name, text, summary, summary2, sent_relas, entities2sents, entities2entities, summary_label, entities, 
         entity_labels, entities2mentions) = input_gen.next() # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      example = Example(data_name, text, summary, summary2, sent_relas, entities2sents, entities2entities, summary_label, entities, 
                        entity_labels, entities2mentions, self._word_vocab, self._entity_vocab, self._hps) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      #if self._hps.mode != 'decode':
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.sent_num) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
  

        batches = []
        for i in xrange(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._hps, self._word_vocab))

      #else: # beam search decode mode
      #  ex = self._example_queue.get()
      #  b = [ex for _ in xrange(self._hps.batch_size)]
      #  self._batch_queue.put(Batch(b, self._hps, self._word_vocab))


  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    """Generates article and abstract text from tf.Example.

    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
      e = example_generator.next() # e is a tf.Example
      try:
        data_name = e.features.feature['data_name'].bytes_list.value[0]
        text = e.features.feature['text'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        summary = e.features.feature['summary'].bytes_list.value[0]
        summary2 = e.features.feature['summary2'].bytes_list.value[0]
        #print(summary2)
        sent_relas = e.features.feature['sent_relas'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        entities2sents = e.features.feature['entities2sents'].bytes_list.value[0]
        entities2entities = e.features.feature['entities2entities'].bytes_list.value[0]
        summary_label = e.features.feature['summary_label'].bytes_list.value[0]
        entities = e.features.feature['entities'].bytes_list.value[0]
        entity_labels = e.features.feature['entity_labels'].bytes_list.value[0]
        entities2mentions = e.features.feature['entities2mentions'].bytes_list.value[0]
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      elif len(entities)==0:
        tf.logging.warning('Found an example with empty entities. Skipping it.')
      else:
        yield (data_name, text, summary, summary2, sent_relas, entities2sents, entities2entities, summary_label, entities, 
               entity_labels, entities2mentions)





