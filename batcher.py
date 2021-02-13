# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

from queue import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data

np_float64=np.float32


class Example(object):

  def __init__(self, data_name, summary_ext, summary, entities, entities2mentions, word_vocab, entity_vocab, hps):

    self.hps = hps

    # Get ids of special tokens
    start_decoding = word_vocab.word2id(data.START_DECODING)
    stop_decoding = word_vocab.word2id(data.STOP_DECODING)
    mention_seg = word_vocab.word2id(data.MENTION_SEG)
    #print(data_name)
    # Process the article
    summary_ext_words = summary_ext.split()
    if len(summary_ext_words) > hps.max_enc_steps:
      summary_ext_words = summary_ext_words[:hps.max_enc_steps]
    self.enc_len = len(summary_ext_words) # store the length after truncation but before padding
    self.enc_input = [word_vocab.word2id(w) for w in summary_ext_words] # list of word ids; OOVs are represented by the id for UNK token

    self.data_name = data_name

    # Process the abstract
    summary_words = summary.split() # list of strings
    summary_ids = [word_vocab.word2id(w) for w in summary_words] # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(summary_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)
    
    entities = entities.split("|||")
    self.entity_num = len(entities)
    self.entity_ids = [entity_vocab.word2id(w) for w in entities]
    
    self.mention_lens = []
    self.mention_ids = []
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
        self.mention_lens.append(len(m2))
        if self.max_mention_len<len(m2):self.max_mention_len=len(m2)

    # If using pointer-generator mode, we need to store some extra info
    if hps.pointer_gen:
      # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
      self.enc_input_extend_vocab, self.article_oovs = data.article2ids(summary_ext_words, word_vocab)

      # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
      extend_vocab = data.abstract2ids(summary_ext_words, word_vocab, self.article_oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      _, self.target = self.get_dec_inp_targ_seqs(extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    self.original_summary_ext = summary_ext
    self.original_summary = summary


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer

    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


  def pad_encoder_input(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    if self.hps.pointer_gen:
      while len(self.enc_input_extend_vocab) < max_len:
        self.enc_input_extend_vocab.append(pad_id)
        
  def pad_entity_ids(self, max_entity_num, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.entity_ids) < max_entity_num:
      self.entity_ids.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, word_vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    if example_list is None: return
    
    self.pad_id = word_vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
    self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings

  def init_encoder_seq(self, example_list, hps):
    max_data_name_len = max([len(ex.data_name) for ex in example_list])
    
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    max_entity_num = max([ex.entity_num for ex in example_list])
    max_mention_len = max([ex.max_mention_len for ex in example_list])

    self.max_entity_num = max_entity_num
    self.max_mention_len = max_mention_len

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
      ex.pad_entity_ids(max_entity_num, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.data_name_batch = np.zeros((hps.batch_size, 20), dtype=np.int32)
    self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
    self.entity_batch = np.zeros((hps.batch_size, max_entity_num), dtype=np.int32)
    self.mention_batch = np.zeros((hps.batch_size, max_entity_num, max_mention_len), dtype=np.int32)
    self.mention_batch.fill(self.pad_id)
    
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np_float64)
    
    self.entity_nums = np.zeros((hps.batch_size), dtype=np.int32)
    self.mention_mask = np.zeros((hps.batch_size, max_entity_num, max_mention_len), dtype=np_float64)
    self.mention_lens = np.zeros((hps.batch_size, max_entity_num), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.data_name_batch[i,:]=[ord(x) for x in ex.data_name[:20]]
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      if (ex.enc_len==0):print("enc_len=0")
      for j in xrange(ex.enc_len):
        self.enc_padding_mask[i][j] = 1.0
      
      self.entity_batch[i, :] = ex.entity_ids[:]
      self.mention_lens[i, :ex.entity_num] = ex.mention_lens[:]
      
      try:
          for I in range(len(ex.mention_ids)):
              for j in range(len(ex.mention_ids[I])):
                    self.mention_batch[i,I,j]=ex.mention_ids[I][j]
      except:
          print("mention_ids:"+ex.data_name)
      
      #self.mention_occur_batch[i, :] = ex.mention_occurs[:]
      self.entity_nums[i] = ex.entity_num
      
      for I in xrange(len(ex.mention_lens)):
        for j in xrange(ex.mention_lens[I]):
            self.mention_mask[i,I,j]=1.0

    # For pointer-generator mode, need to store some extra info
    if hps.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list, hps):
    """Initializes the following:
        self.dec_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
        self.target_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
        self.dec_padding_mask:
          numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
        """
    # Pad the inputs and targets
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
        
    #print("target:", self.dec_batch)
  def concat(self, batch2):
    batch=Batch(None,None,None)
    
    batch.data_name_batch = np.concatenate([self.data_name_batch, batch2.data_name_batch], axis=0)
    batch.dec_batch = np.concatenate([self.dec_batch, batch2.dec_batch], axis=0)
    batch.target_batch = np.concatenate([self.target_batch, batch2.target_batch], axis=0)
    batch.dec_padding_mask = np.concatenate([self.dec_padding_mask, batch2.dec_padding_mask], axis=0)
    
    batch.entity_nums = np.concatenate([self.entity_nums, batch2.entity_nums], axis=0)
    batch.entity_batch = np.concatenate([self.entity_batch, batch2.entity_batch], axis=0)
    batch.mention_batch = np.concatenate([self.mention_batch, batch2.mention_batch], axis=0)
    batch.mention_mask = np.concatenate([self.mention_mask, batch2.mention_mask], axis=0)
    
    #max_enc_len=max(np.shape(self.enc_batch)[1], np.shape(batch2.enc_batch)[1])
    #batch.enc_batch
    batch.enc_batch = np.concatenate([self.enc_batch, batch2.enc_batch], axis=0)
    batch.enc_lens = np.concatenate([self.enc_lens, batch2.enc_lens], axis=0)
    batch.enc_padding_mask = np.concatenate([self.enc_padding_mask, batch2.enc_padding_mask], axis=0)
    
    batch.max_art_oovs = self.max_art_oovs
    batch.art_oovs = []
    batch.art_oovs.extend(self.art_oovs)
    batch.art_oovs.extend(batch2.art_oovs)
    batch.enc_batch_extend_vocab = np.concatenate([self.enc_batch_extend_vocab, batch2.enc_batch_extend_vocab], axis=0)
    
    return batch

  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""
    self.original_summary_ext = [ex.original_summary_ext for ex in example_list] # list of lists
    self.original_summary = [ex.original_summary for ex in example_list] # list of lists
    #self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


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
        (data_name, summary_ext, summary, entities, entities2mentions) = input_gen.next() # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      example = Example(data_name, summary_ext, summary, entities, entities2mentions, self._word_vocab, self._entity_vocab, self._hps) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      if self._hps.mode != 'decode' or self._hps.dec_method!="beam-search":
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in xrange(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._hps, self._word_vocab))

      else: # beam search decode mode
        ex = self._example_queue.get()
        b = [ex for _ in xrange(self._hps.batch_size)]
        self._batch_queue.put(Batch(b, self._hps, self._word_vocab))


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
        summary_ext = e.features.feature['summary_ext'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        summary = e.features.feature['summary'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        entities = e.features.feature['entities'].bytes_list.value[0]
        entities2mentions = e.features.feature['entities2mentions'].bytes_list.value[0]
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(summary_ext)==0: # See https://github.com/abisee/pointer-generator/issues/1
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      elif len(entities)==0:
        tf.logging.warning('Found an example with empty entities. Skipping it.')
      else:
        yield (data_name, summary_ext, summary, entities, entities2mentions)






