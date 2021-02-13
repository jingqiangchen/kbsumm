
"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
import logging
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from sklearn.metrics import precision_score, recall_score, f1_score

import rouge
evaluator = rouge.Rouge(metrics=['rouge-n', "rouge-l"],
                               max_n=2,
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, word_vocab, entity_vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    # Make a descriptive decode directory name
    #ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
    #self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
    self._decode_dir = os.path.join(FLAGS.log_root, "decode-"+FLAGS.corpus)
    #if os.path.exists(self._decode_dir):
    #  raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

    # Make the decode dir if necessary
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

      # Make the dirs to contain output written in the correct format for pyrouge
    self._output_file = os.path.join(self._decode_dir, ckpt_path.split('-')[-1])
    #self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
    #if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
    #self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
    #if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
    self.score_1=[0.,0.,0.]
    self.score_2=[0.,0.,0.]
    self.score_l=[0.,0.,0.]
    self.num=0.

  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    output_file = open(self._output_file, "w")
        
    attn_file = open(self._output_file+"-attn", "w")
    
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        #tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        #results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        #rouge_log(results_dict, self._decode_dir)
        output_file.close()
        return

      original_summary_ext = batch.original_summary_ext[0]  # string
      original_summary = batch.original_summary[0]  # string

      summary_ext_withunks = data.show_art_oovs(original_summary_ext, self._word_vocab) # string
      summary_withunks = data.show_abs_oovs(original_summary, self._word_vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get best Hypothesis
      best_hyp = beam_search.run_beam_search(self._sess, self._model, self._word_vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
      output_ids = [int(t) for t in best_hyp.tokens[1:]]
      decoded_words = data.outputids2words(output_ids, self._word_vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
      
      # Remove the [STOP] token from decoded_words, if necessary
      try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
      except ValueError:
        decoded_words = decoded_words
    
      try:
        fst_pad_idx = decoded_words.index(data.PAD_TOKEN)
        decoded_words = decoded_words[:fst_pad_idx]
      except ValueError:
        decoded_words = decoded_words
        
      decoded_output = ' '.join(decoded_words) # single string
      

      if FLAGS.single_pass:
        self.write_results(original_summary, decoded_words, best_hyp.attn_dists, counter, output_file, attn_file) # write ref summary and decoded summary to file, to eval with pyrouge later
        counter += 1 # this is how many examples we've decoded
      else:
        print_results(summary_ext_withunks, summary_withunks, decoded_output) # log output to screen
        self.write_for_attnvis(summary_ext_withunks, summary_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

        # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
        t1 = time.time()
        if t1-t0 > SECS_UNTIL_NEW_CKPT:
          tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
          _ = util.load_ckpt(self._saver, self._sess)
          t0 = time.time()
    output_file.close()

  def write_results(self, reference_summary, decoded_words, attn_dist, ex_index, output_file, attn_file=None):

    '''decoded_sents = []
    #print("decoded_words:%s\n" % decoded_words)
    
    while len(decoded_words) > 0:
      try:
        fst_period_idx = decoded_words.index(".")
      except ValueError: # there is text remaining that doesn't end in "."
        fst_period_idx = len(decoded_words)
      sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
      decoded_words = decoded_words[fst_period_idx+1:] # everything else
      decoded_sents.append(' '.join(sent))
    '''

    decoded_text = " ".join(decoded_words)
    #print("decoded_words:%s\n" % decoded_text)

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    #decoded_text = [make_html_safe(w) for w in decoded_sents]
    #reference_caption = [make_html_safe(w) for w in reference_caption]

    # Write to file
    target = reference_summary.strip("\n")
    summary = decoded_text.strip("\n")
    ex_index += 1
    output_file.write("[%d]\n" % ex_index)
    output_file.write('---------------------------summary----------------\n')
    output_file.write(summary+"\n")
    output_file.write('------------------target---summary----------------\n')
    output_file.write(target+'\n')
    
    try:
        output_file.write('-------------------summary-scores---------------\n')
          
        scores = evaluator.get_scores([reference_summary], [decoded_text])
        output_file.write("%f\t%f\t%f\n" % (scores[0]['rouge-1']['r'], scores[0]['rouge-1']['p'], scores[0]['rouge-1']['f']))
        self.score_1[0]+=float(scores[0]['rouge-1']['r'])
        self.score_1[1]+=float(scores[0]['rouge-1']['p'])
        self.score_1[2]+=float(scores[0]['rouge-1']['f'])
        output_file.write("%f\t%f\t%f\n" % (self.score_1[0]/ex_index, self.score_1[1]/ex_index, self.score_1[2]/ex_index))
          
        output_file.write("%f\t%f\t%f\n" % (scores[0]['rouge-2']['r'], scores[0]['rouge-2']['p'], scores[0]['rouge-2']['f']))
        self.score_2[0]+=float(scores[0]['rouge-2']['r'])
        self.score_2[1]+=float(scores[0]['rouge-2']['p'])
        self.score_2[2]+=float(scores[0]['rouge-2']['f'])
        output_file.write("%f\t%f\t%f\n" % (self.score_2[0]/ex_index, self.score_2[1]/ex_index, self.score_2[2]/ex_index))
          
        output_file.write("%f\t%f\t%f\n" % (scores[0]['rouge-l']['r'], scores[0]['rouge-l']['p'], scores[0]['rouge-l']['f']))
        self.score_l[0]+=float(scores[0]['rouge-l']['r'])
        self.score_l[1]+=float(scores[0]['rouge-l']['p'])
        self.score_l[2]+=float(scores[0]['rouge-l']['f'])
        output_file.write("%f\t%f\t%f\n" % (self.score_l[0]/ex_index, self.score_l[1]/ex_index, self.score_l[2]/ex_index))
        
        output_file.write('-------------------------------------\n')
        
        if attn_file is not None:
            attn_file.write("%d %d\n" % (np.shape(attn_dist)[0], np.shape(attn_dist)[1]))
            str_attn="\n".join([" ".join(["%.4f" % one_attn for one_attn in attn]) for attn in attn_dist])
            attn_file.write(str_attn + "\n")
            
    except Exception as e:
        output_file.write(str(e)+"\n")
        output_file.write("ERROR!\n")


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


class GreedyDecoder(object):

  def __init__(self, model, batcher, word_vocab, entity_vocab):
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    self._decode_dir = os.path.join(FLAGS.log_root, "greedy-"+FLAGS.corpus)
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

    self._output_file = os.path.join(self._decode_dir, ckpt_path.split('-')[-1])
    self.score_1=[0.,0.,0.]
    self.score_2=[0.,0.,0.]
    self.score_l=[0.,0.,0.]
    self.num=0.

  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    output_file = open(self._output_file, "w")
    
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        #tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        #results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        #rouge_log(results_dict, self._decode_dir)
        output_file.close()
        return

      hyps, lens = beam_search.run_greedy_search(self._sess, self._model, self._word_vocab, batch)

      # Extract the output ids from the hypothesis and convert back to words
      decoded_outputs = []
      #for hyp in hyps:
      for i in range(len(hyps)):
        
        hyp = hyps[i]
        output_ids = [int(t) for t in hyp.tokens[1:lens[i]]]
        decoded_words = data.outputids2words(output_ids, self._word_vocab, (batch.art_oovs[i] if FLAGS.pointer_gen else None))
          
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
    
        try:
            fst_pad_idx = decoded_words.index(data.PAD_TOKEN)
            decoded_words = decoded_words[:fst_pad_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words) # single string
        self.write_results(batch.original_summary[i], decoded_output, counter*FLAGS.batch_size+i, output_file)
      
      output_file.flush()
      counter+=1
    
    output_file.close()

  def write_results(self, reference_summary, decoded_text, ex_index, output_file):

    #decoded_text = " ".join(decoded_words)

    # Write to file
    target = reference_summary.strip("\n")
    summary = decoded_text.strip("\n")
    ex_index += 1
    output_file.write("[%d]\n" % ex_index)
    output_file.write('---------------------------summary----------------\n')
    output_file.write(summary+"\n")
    output_file.write('------------------target---summary----------------\n')
    output_file.write(target+'\n')
    
    try:
        output_file.write('-------------------summary-scores---------------\n')
          
        scores = evaluator.get_scores([reference_summary], [decoded_text])
        output_file.write("%f\t%f\t%f\n" % (scores[0]['rouge-1']['r'], scores[0]['rouge-1']['p'], scores[0]['rouge-1']['f']))
        self.score_1[0]+=float(scores[0]['rouge-1']['r'])
        self.score_1[1]+=float(scores[0]['rouge-1']['p'])
        self.score_1[2]+=float(scores[0]['rouge-1']['f'])
        output_file.write("%f\t%f\t%f\n" % (self.score_1[0]/ex_index, self.score_1[1]/ex_index, self.score_1[2]/ex_index))
          
        output_file.write("%f\t%f\t%f\n" % (scores[0]['rouge-2']['r'], scores[0]['rouge-2']['p'], scores[0]['rouge-2']['f']))
        self.score_2[0]+=float(scores[0]['rouge-2']['r'])
        self.score_2[1]+=float(scores[0]['rouge-2']['p'])
        self.score_2[2]+=float(scores[0]['rouge-2']['f'])
        output_file.write("%f\t%f\t%f\n" % (self.score_2[0]/ex_index, self.score_2[1]/ex_index, self.score_2[2]/ex_index))
          
        output_file.write("%f\t%f\t%f\n" % (scores[0]['rouge-l']['r'], scores[0]['rouge-l']['p'], scores[0]['rouge-l']['f']))
        self.score_l[0]+=float(scores[0]['rouge-l']['r'])
        self.score_l[1]+=float(scores[0]['rouge-l']['p'])
        self.score_l[2]+=float(scores[0]['rouge-l']['f'])
        output_file.write("%f\t%f\t%f\n" % (self.score_l[0]/ex_index, self.score_l[1]/ex_index, self.score_l[2]/ex_index))
        
        output_file.write('-------------------------------------\n')
            
    except Exception as e:
        output_file.write(str(e)+"\n")
        output_file.write("ERROR!\n")


  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file)
    tf.logging.info('Wrote visualization data to %s', output_fname)


class BeamSearchDecoder_RL(object):
  """Beam search decoder."""

  def __init__(self, model, word_vocab, entity_vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt_gen(self._saver, self._sess)

  def decode(self, batch):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0

    #original_summary_ext = batch.original_summary_ext[0]  # string
    #original_summary = batch.original_summary[0]  # string

    #summary_ext_withunks = data.show_art_oovs(original_summary_ext, self._word_vocab) # string
    #summary_withunks = data.show_abs_oovs(original_summary, self._word_vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

    # Run beam search to get best Hypothesis
    best_hyp = beam_search.run_beam_search(self._sess, self._model, self._word_vocab, batch)

    # Extract the output ids from the hypothesis and convert back to words
    output_ids = [int(t) for t in best_hyp.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, self._word_vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))
      
    # Remove the [STOP] token from decoded_words, if necessary
    try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
    except ValueError:
        decoded_words = decoded_words
    decoded_output = ' '.join(decoded_words) # single string
    
    return decoded_output 


class GreedyDecoder_RL(object):
  """Greedy decoder."""

  def __init__(self, model, word_vocab, entity_vocab):

    self._model = model
    self.frozen_vars = self._model.build_graph()
    #self._model.build_graph()
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt_gen(self._saver, self._sess)

  def decode(self, batch):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0

    hyps, lens = beam_search.run_greedy_search(self._sess, self._model, self._word_vocab, batch)

    # Extract the output ids from the hypothesis and convert back to words
    decoded_outputs = []
    #for hyp in hyps:
    for i in range(len(hyps)):
        
        hyp = hyps[i]
        output_ids = [int(t) for t in hyp.tokens[1:lens[i]]]
        decoded_words = data.outputids2words(output_ids, self._word_vocab, (batch.art_oovs[i] if FLAGS.pointer_gen else None))
          
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
            #print(fst_stop_idx)
            if fst_stop_idx == 0:
                decoded_words = [data.STOP_DECODING]
            else:
                decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words) # single string
        decoded_outputs.append(decoded_output)
    
    return decoded_outputs 


class GreedyDecoder_RL_2(object):
  """Greedy decoder."""

  def __init__(self, model, word_vocab, entity_vocab):

    self._model = model
    self._model.build_graph()
    self._word_vocab = word_vocab
    self._entity_vocab = entity_vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt_gen(self._saver, self._sess)

  def decode(self, batch):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0

    decoded_ids = beam_search.run_greedy_search_2(self._sess, self._model, self._word_vocab, batch)
    decoded_ids = np.array(decoded_ids)
    decoded_ids = np.transpose(decoded_ids, [1, 0]).tolist()

    # Extract the output ids from the hypothesis and convert back to words
    decoded_outputs = []
    #for hyp in hyps:
    for i in range(FLAGS.batch_size):
        
        decoded_id = decoded_ids[i]
        output_ids = [int(t) for t in decoded_id[:]]
        decoded_words = data.outputids2words(output_ids, self._word_vocab, (batch.art_oovs[i] if FLAGS.pointer_gen else None))
          
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
            #print(fst_stop_idx)
            if fst_stop_idx == 0:
                decoded_words = [data.STOP_DECODING]
            else:
                decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words) # single string
        decoded_outputs.append(decoded_output)
    
    return decoded_outputs 


class ExtGreedyDecoder(object):

  def __init__(self, model, batcher):
    
    self._model = model
    
    self._batcher = batcher
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    ckpt_path = util.load_ckpt(self._saver, self._sess)

    self._decode_dir = os.path.join(FLAGS.log_root, "ext_greedy_decode-"+FLAGS.corpus)

    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

      # Make the dirs to contain output written in the correct format for pyrouge
    self._output_file = os.path.join(self._decode_dir, ckpt_path.split('-')[-1])

    self.num=0.
    self.p=[0.0] * 10
    self.r=[0.0] * 10
    self.f1score=[0.0] * 10
    
    self.score_1=np.zeros([10,3],dtype=np.float32).tolist()
    self.score_l=np.zeros([10,3],dtype=np.float32).tolist()
    self.score_2=np.zeros([10,3],dtype=np.float32).tolist()

  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    output_file = open(self._output_file, "w")
    
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        #tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        output_file.close()
        return

      sent_scores, entity_scores, data_name = self._model.greedy_decode(self._sess, batch)
      
      self.write_results(sent_scores, entity_scores, data_name, batch, counter, output_file) 
      counter += 1 

      t1 = time.time()
      if t1-t0 > SECS_UNTIL_NEW_CKPT:
        tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
        _ = util.load_ckpt(self._saver, self._sess)
        t0 = time.time()
    
    output_file.close()

  def write_results(self, sent_scores, entity_scores, data_name, batch, counter, output_file):
    batch_size=len(sent_scores)
    for i in range(batch_size):
        self.num+=1
        
        ext_sent = []
        sent_num = batch.sent_nums[i]
        for _ in range(sent_num-1):
            ext_sent.append(0)
        sent_score = sent_scores[i].tolist()
        stop_score = sent_score[sent_num-1]
        for j in range(len(sent_score)):
            if sent_score[j] > stop_score:
                ext_sent[j]=1
        
        label=[]
        sent_label=batch.sent_label_batch[i].tolist()
        for j in range(sent_num-1):
            if sent_label[j]>0:
                label.append(1)
            else:
                label.append(0)

        output_file.write("[%d]\n" % (counter * batch_size + i))
        output_file.write('---------------------------sent_scores------------------------\n')
        output_file.write("\t".join(["%.2f" % item for item in sent_score])+"\n")
        output_file.write('---------------------------sent_labels-----        -----------\n')
        output_file.write("\t".join(["%.2f" % item for item in sent_label])+"\n")
        output_file.write('---------------------------sent_labels------------------------\n')
        #print(sent_label, sent_score)
        p = precision_score(label, ext_sent, average='binary')
        r = recall_score(label, ext_sent, average='binary')
        f1score = f1_score(label, ext_sent, average='binary')
        self.p[0]+=p
        self.r[0]+=r
        self.f1score[0]+=f1score
        output_file.write("0-precision:%.2f\trecall:%.2f\tf1-measure:%.2f\n" % (p, r, f1score))
        output_file.write("0-precision:%.2f\trecall:%.2f\tf1-measure:%.2f\n" % (self.p[0]/self.num, self.r[0]/self.num, self.f1score[0]/self.num))
        
        sent_score = sent_scores[i][:-1]
        ix, = np.unravel_index(sent_score.argsort(axis=0), dims=sent_score.shape)
        ix = ix[::-1]
        for k in range(3,6):
            ext_summary = ""
            ext_sent=[]
            a_ix = ix[:k]
            for j in range(sent_num-1):
                if j in a_ix:
                    ext_sent.append(1)
                    ext_summary += batch.sent_str_batch[i][j]+" "
                else:
                    ext_sent.append(0)
            p = precision_score(label, ext_sent, average='binary')
            r = recall_score(label, ext_sent, average='binary')
            f1score = f1_score(label, ext_sent, average='binary')
            self.p[k]+=p
            self.r[k]+=r
            self.f1score[k]+=f1score
            output_file.write("%d-precision:%.2f\trecall:%.2f\tf1-measure:%.2f\n" % (k, p, r, f1score))
            output_file.write("%d-precision:%.2f\trecall:%.2f\tf1-measure:%.2f\n" % (k, self.p[k]/self.num, self.r[k]/self.num, self.f1score[k]/self.num))
            output_file.write("%s\n" % ext_summary)
            output_file.write("%s\n" % batch.summary_str_batch[i])
        
            scores = evaluator.get_scores([batch.summary_str_batch[i]], [ext_summary.strip()])
            output_file.write("%d-%f\t%f\t%f\n" % (k, scores[0]['rouge-1']['r'], scores[0]['rouge-1']['p'], scores[0]['rouge-1']['f']))
            self.score_1[k][0]+=float(scores[0]['rouge-1']['r'])
            self.score_1[k][1]+=float(scores[0]['rouge-1']['p'])
            self.score_1[k][2]+=float(scores[0]['rouge-1']['f'])
            output_file.write("%d-%f\t%f\t%f\n" % (k, self.score_1[k][0]/self.num, self.score_1[k][1]/self.num, self.score_1[k][2]/self.num))
            
            output_file.write("%d-%f\t%f\t%f\n" % (k, scores[0]['rouge-l']['r'], scores[0]['rouge-l']['p'], scores[0]['rouge-l']['f']))
            self.score_l[k][0]+=float(scores[0]['rouge-l']['r'])
            self.score_l[k][1]+=float(scores[0]['rouge-l']['p'])
            self.score_l[k][2]+=float(scores[0]['rouge-l']['f'])
            output_file.write("%d-%f\t%f\t%f\n" % (k, self.score_l[k][0]/self.num, self.score_l[k][1]/self.num, self.score_l[k][2]/self.num))
            
            output_file.write("%d-%f\t%f\t%f\n" % (k, scores[0]['rouge-2']['r'], scores[0]['rouge-2']['p'], scores[0]['rouge-2']['f']))
            self.score_2[k][0]+=float(scores[0]['rouge-2']['r'])
            self.score_2[k][1]+=float(scores[0]['rouge-2']['p'])
            self.score_2[k][2]+=float(scores[0]['rouge-2']['f'])
            output_file.write("%d-%f\t%f\t%f\n" % (k, self.score_2[k][0]/self.num, self.score_2[k][1]/self.num, self.score_2[k][2]/self.num))


class ExtGenDecoder(object):

  def __init__(self, model_ext, model_gen, batcher, word_vocab, mode="decode"):
    self._batcher = batcher
    self._word_vocab = word_vocab
    self._mode = mode
    
    self._model_gen = model_gen
    if model_gen is not None:
        self._model_gen.build_graph()
        self._saver_gen = tf.train.Saver() # we use this to load checkpoints for decoding
        self._sess_gen = tf.Session(config=util.get_config())
        ckpt_path_gen = util.load_ckpt_gen(self._saver_gen, self._sess_gen)
    
    self._model_ext = model_ext
    self._model_ext.build_graph()
    self._saver_ext = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess_ext = tf.Session(config=util.get_config())
    ckpt_path_ext = util.load_ckpt(self._saver_ext, self._sess_ext)
    
    if mode == "decode":
        self._decode_dir = os.path.join(FLAGS.log_root, "ext-gen-"+FLAGS.corpus)
    else:
        self._decode_dir = os.path.join(FLAGS.log_root, "train-ext-gen-"+FLAGS.corpus)
    
    if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

      # Make the dirs to contain output written in the correct format for pyrouge
    self._ext_output_file = os.path.join(self._decode_dir, ckpt_path_ext.split('-')[-1] + ".ex")
    self._gen_output_file = os.path.join(self._decode_dir, ckpt_path_ext.split('-')[-1] + ".ge")
    self._trn_output_file = os.path.join(self._decode_dir, ckpt_path_ext.split('-')[-1] + ".tn")

  def decode(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    
    if self._mode == "decode":
        self.decode1()
    else:
        self.decode2()
    

  def decode1(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    
    ext_output_file = open(self._ext_output_file, "w")
    if self._model_gen is not None:
        gen_output_file = open(self._gen_output_file, "w")
    
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        #tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        ext_output_file.close()
        return

      #ext
      sent_scores, entity_scores, data_name = self._model_ext.greedy_decode(self._sess_ext, batch)
      
      self.write_results_ext(sent_scores, entity_scores, data_name, batch, counter, ext_output_file) 
      
      #gen
      if self._model_gen is not None:
          for index in range(FLAGS.batch_size):
            sent_score = sent_scores[index]
            sent_indices, = np.unravel_index(sent_score.argsort(axis=0), dims=sent_score.shape)
            sent_indices = sent_indices[::-1]
            sent_indices = sent_indices[:FLAGS.sample_sent_num]
            sent_indices.sort()
            #print(sampled_sent_indices[index])
                  
            sents = batch.sent_str_batch[index]
            summary_ext = []
            #print("sent_indices:", sent_indices)
            #print("sent_num:", len(sents))
            sent_num = len(sents)
            for sent_index in sent_indices:
                if sent_index < sent_num:
                    summary_ext.append(sents[sent_index])
            summary_ext = " ".join(summary_ext)
                  
            entity_num = batch.entity_nums[index]
            entity_score = entity_scores[index]
            entity_indices, = np.unravel_index(entity_score.argsort(axis=0), dims=entity_score.shape)
            entity_indices = entity_indices[::-1]
            entity_indices = entity_indices[:FLAGS.sample_entity_num]
            entity_indices = np.array(entity_indices)
            entity_indices.sort()
                  
            #index, reps, summary_ext, max_enc_steps, entity_indices
            gen_batch = batch.tile_for_gen(index, FLAGS.beam_size, summary_ext, FLAGS.max_enc_steps, entity_indices)
            
            best_hyp = beam_search.run_beam_search(self._sess_gen, self._model_gen, self._word_vocab, gen_batch)
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._word_vocab, (gen_batch.art_oovs[0] if FLAGS.pointer_gen else None))
          
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            gen_output = ' '.join(decoded_words) # single string
            
            self.write_results_gen(batch.summary_str_batch[index], batch.summary_str_batch2[index], gen_output, counter*FLAGS.batch_size+index, gen_output_file)
      
      counter += 1 

      t1 = time.time()
      if t1-t0 > SECS_UNTIL_NEW_CKPT:
        #tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
        #_ = util.load_ckpt_gen(self._saver_gen, self._sess_gen)
        #_ = util.load_ckpt(self._saver_ext, self._sess_ext)
        t0 = time.time()
    
    ext_output_file.close()
    if self._model_gen is not None:
        gen_output_file.close()
    
  
  def decode2(self):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    
    trn_output_file = open(self._trn_output_file, "w")
    
    while True:
      batch = self._batcher.next_batch()  # 1 example repeated across batch
      
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        #tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir, self._rouge_dec_dir)
        trn_output_file.close()
        return

      sent_scores, entity_scores, data_name = self._model_ext.greedy_decode(self._sess_ext, batch)
      
      all_summary_ext_greedy = []
      all_entity_indices_greedy = []
      for index in range(FLAGS.batch_size):
        #sampler
        sent_num = batch.sent_nums[index]
        entity_num = batch.entity_nums[index]
        
        sent_score = sent_scores[index]
        sent_indices, = np.unravel_index(sent_score.argsort(axis=0), dims=sent_score.shape)
        sent_indices = sent_indices[::-1]
        sent_indices = sent_indices[:FLAGS.sample_sent_num]
        sent_indices.sort()
        sents = batch.sent_str_batch[index]
        summary_ext = []
        for sent_index in sent_indices:
            if sent_index < sent_num:
                summary_ext.append(sents[sent_index])
        summary_ext = " ".join(summary_ext)
        all_summary_ext_greedy.append(summary_ext)
        
        entity_score = entity_scores[index]
        a_entity_indices, = np.unravel_index(entity_score.argsort(axis=0), dims=entity_score.shape)
        a_entity_indices = a_entity_indices[::-1]
        a_entity_indices = a_entity_indices[:FLAGS.sample_entity_num]
        a_entity_indices = np.array(a_entity_indices)
        entity_indices = []
        for entity_index in a_entity_indices:
            if entity_index < entity_num:
                entity_indices.append(entity_index)
        entity_indices = np.array(entity_indices)
        all_entity_indices_greedy.append(entity_indices)

      gen_batch_greedy = batch.tile_for_greedy(all_summary_ext_greedy, FLAGS.max_enc_steps, all_entity_indices_greedy)
      hyps, lens = beam_search.run_greedy_search(self._sess_gen, self._model_gen, self._word_vocab, gen_batch_greedy)

      # Extract the output ids from the hypothesis and convert back to words
      decoded_outputs = []
      #for hyp in hyps:
      for i in range(len(hyps)):
        
        hyp = hyps[i]
        output_ids = [int(t) for t in hyp.tokens[1:lens[i]]]
        decoded_words = data.outputids2words(output_ids, self._word_vocab, (gen_batch_greedy.art_oovs[i] if FLAGS.pointer_gen else None))
          
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
            #print(fst_stop_idx)
            if fst_stop_idx == 0:
                decoded_words = [data.STOP_DECODING]
            else:
                decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words) # single string
        self.write_results_gen_for_train(batch.summary_str_batch[i], batch.summary_str_batch2[i], decoded_output, all_summary_ext_greedy[i], counter*FLAGS.batch_size+i, trn_output_file)
      
      trn_output_file.flush()
    
    trn_output_file.close()
          

  def write_results_ext(self, sent_scores, entity_scores, data_name, batch, counter, output_file):
    batch_size=len(sent_scores)
    for i in range(batch_size):
        ext_sent = []
        sent_num = batch.sent_nums[i]
        for _ in range(sent_num-1):
            ext_sent.append(0)
        sent_score = sent_scores[i].tolist()
        stop_score = sent_score[sent_num-1]
        for j in range(len(sent_score)):
            if sent_score[j] > stop_score:
                ext_sent[j]=1
        
        label=[]
        sent_label=batch.sent_label_batch[i].tolist()
        for j in range(sent_num-1):
            if sent_label[j]>0:
                label.append(1)
            else:
                label.append(0)

        sent_score = sent_scores[i][:-1]
        ix, = np.unravel_index(sent_score.argsort(axis=0), dims=sent_score.shape)
        ix = ix[::-1]
        for k in range(4,5):
            ext_summary = ""
            ext_sent=[]
            a_ix = ix[:k]
            for j in range(sent_num-1):
                if j in a_ix:
                    ext_sent.append(1)
                    ext_summary += batch.sent_str_batch[i][j]+" "
                else:
                    ext_sent.append(0)
            output_file.write("%s\n" % ext_summary)


  def write_results_gen(self, ref, ref2, decoded_text, ex_index, output_file):
    target = ref2.strip("\n")
    summary = decoded_text.strip("\n")
    output_file.write(summary+"\n")
    output_file.write(target+'\n')
        
  
  def write_results_gen_for_train(self, ref, ref2, decoded_text, summary_ext, ex_index, output_file):
    target = ref.strip("\n")
    target2 = ref2.strip("\n")
    summary = decoded_text.strip("\n")
    summary_ext = summary_ext.strip("\n")
    '''
    try:
        scores = rouge.get_scores([target], [summary])
        scores_ext = rouge.get_scores([target], [summary_ext])
    except:
        scores = rouge.get_scores([" ".join(target.split()[:FLAGS.max_dec_steps])], [" ".join(decoded_text.split()[:FLAGS.max_dec_steps])])
        scores_ext = rouge.get_scores([" ".join(target.split()[:FLAGS.max_dec_steps])], [" ".join(summary_ext.split()[:FLAGS.max_dec_steps])])
    print(scores)
    print(scores_ext)
    output_file.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f'],
                                                    scores_ext[0]['rouge-1']['f'], scores_ext[0]['rouge-2']['f'], scores_ext[0]['rouge-l']['f']))
    '''
    output_file.write(summary+"\n")
    output_file.write(summary_ext+"\n")
    output_file.write(target2+"\n")
    output_file.write(target+"\n")


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print ("")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print ("")


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  tf.logging.info(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname












