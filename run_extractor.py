
"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher_gcn import Batcher
from model_gcn import SummarizationModel_GCN
from model import SummarizationModel
from decode3 import BeamSearchDecoder_RL, ExtGreedyDecoder, GreedyDecoder_RL, GreedyDecoder_RL_2, ExtGenDecoder
import util
from tensorflow.python import debug as tf_debug
import common

from rouge import Rouge
rouge=Rouge()

FLAGS = tf.app.flags.FLAGS

# Where to find data
#tf.app.flags.DEFINE_string('data_path', common.path_chunked_train+"/*", 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('data_path', "/home/test/kbsumm/data/dailymail/chunked/example.bin", 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')

tf.app.flags.DEFINE_string('word_vocab_path', common.path_word_vocab, 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('entity_vocab_path', common.path_entity_vocab, 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('word_embedding_path', common.path_word_vocab_embeddings, 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('entity_embedding_path', common.path_entity_vocab_embeddings, 'Path expression to text vocabulary file.') 

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('dec_method', 'beam-search', 'must be one of beam-search/greedy')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_string('write_mode', 'decode', 'must be one of decide/train')
tf.app.flags.DEFINE_boolean('decode_gen', True, '')

# Where to save output
tf.app.flags.DEFINE_string('log_root', common.path_log_root_gcn, 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'W.O.RL', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('corpus', 'dailymail', '')

# Generation model
tf.app.flags.DEFINE_string('gen_log_root', common.path_log_root, 'Root directory for all logging.')
tf.app.flags.DEFINE_string('gen_exp_name', 'TEO', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')

tf.app.flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Hyperparameters
#tf.app.flags.DEFINE_integer('max_entity_num', 50, 'number of image features')
tf.app.flags.DEFINE_integer('encoder_hidden_dim', 256, 'dimension of RNN hidden states of Encoder')
tf.app.flags.DEFINE_integer('hidden_dim', 512, 'dimension of RNN hidden states')

tf.app.flags.DEFINE_integer('max_position', 500, 'number of image features')

tf.app.flags.DEFINE_integer('word_emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('entity_emb_dim', 128, 'dimension of word embeddings')

tf.app.flags.DEFINE_integer('batch_size', 10, 'minibatch size')

tf.app.flags.DEFINE_integer('word_vocab_size', 40004, 'Size of word vocabulary.')
tf.app.flags.DEFINE_integer('entity_vocab_size', 100001, 'Size of entity vocabulary.')

tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Entity-regarded parameters
tf.app.flags.DEFINE_boolean('use_entity', True, '')
tf.app.flags.DEFINE_boolean('use_gcn_entity', True, '')
tf.app.flags.DEFINE_boolean('use_mention_occurs', True, '')
tf.app.flags.DEFINE_boolean('use_entity_embedding', True, '')
tf.app.flags.DEFINE_boolean('gen_use_entity_embedding', False, '')
tf.app.flags.DEFINE_float('lambda_ee_train', 0.0, '')

tf.app.flags.DEFINE_integer('gcn_level', 2, '')
tf.app.flags.DEFINE_float('entity_lambda', 0.3, '')
tf.app.flags.DEFINE_float('rl_lambda', 0, '')
tf.app.flags.DEFINE_string('pos_emb_type', 'rnn', '')
tf.app.flags.DEFINE_integer('max_sent_num', 360, '')#168, 160
tf.app.flags.DEFINE_integer('max_entity_num', 1000, '')#168, 160

tf.app.flags.DEFINE_integer('sample_sent_num', 3, '')
tf.app.flags.DEFINE_integer('sample_entity_num', 4, '')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

tf.app.flags.DEFINE_integer('ckpt_max_to_keep', 50, ' ')
tf.app.flags.DEFINE_integer('ckpt_save_model_secs', 5000, ' ')


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):

  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss


def restore_best_model():
  """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
  tf.logging.info("Restoring bestmodel for training...")

  # Initialize all vars in the model
  sess = tf.Session(config=util.get_config())
  print ("Initializing all variables...")
  sess.run(tf.initialize_all_variables())

  # Restore the best model from eval dir
  saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
  print ("Restoring all non-adagrad variables from best model in eval dir...")
  curr_ckpt = util.load_ckpt(saver, sess, "eval")
  print ("Restored %s." % curr_ckpt)

  # Save this model to train dir and quit
  new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
  new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
  print ("Saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
  new_saver.save(sess, new_fname)
  print ("Saved.")
  exit()


def convert_to_coverage_model():
  """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
  tf.logging.info("converting non-coverage model to coverage model..")

  # initialize an entire coverage model from scratch
  sess = tf.Session(config=util.get_config())
  print ("initializing everything...")
  sess.run(tf.global_variables_initializer())

  # load all non-coverage weights from checkpoint
  saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
  print ("restoring non-coverage variables...")
  curr_ckpt = util.load_ckpt(saver, sess)
  print ("restored.")

  # save this model and quit
  new_fname = curr_ckpt + '_cov_init'
  print ("saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this one will save all variables that now exist
  new_saver.save(sess, new_fname)
  print ("saved.")
  exit()


def gen_model(word_vocab, entity_vocab, hps):
  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.gen_log_root = os.path.join(FLAGS.gen_log_root, FLAGS.gen_exp_name)
  
  # If single_pass=True, check we're in decode mode
  if FLAGS.single_pass and FLAGS.mode!='decode':
    raise Exception("The single_pass flag should only be True in decode mode")

  tf.set_random_seed(111) # a seed value for randomness

  decode_model_hps = hps  # This will be the hyperparameters for the decoder model
  decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
  
  model = SummarizationModel(decode_model_hps, word_vocab, entity_vocab)
  if FLAGS.dec_method == "beam-search":
    decoder = BeamSearchDecoder_RL(model, word_vocab, entity_vocab)
  elif FLAGS.dec_method == "greedy":
    decoder = GreedyDecoder_RL(model, word_vocab, entity_vocab)
  
  return decoder

def training_greedy_bak(model,batch,sess,gen_decoder):
    batch.rl_sent_label_batch = np.zeros(np.shape(batch.sent_label_batch), dtype=np.float32)
    batch.rl_entity_label_batch = np.zeros(np.shape(batch.entity_label_batch), dtype=np.float32)
          
    #sampling
    samples = model.sampler(sess, batch)
    sampled_sent_indices = samples["sampled_sents"]
    sampled_entity_indices = samples["sampled_entities"]
    
    sent_scores = samples["sent_scores"]
    entity_scores = samples["entity_scores"]
    
    all_sent_indices_sample = []
    all_summary_ext_sample = []
    all_entity_indices_sample = []
    
    all_summary_ext_greedy = []
    all_entity_indices_greedy = []
    for index in range(FLAGS.batch_size):
        #sampler
        sent_num = batch.sent_nums[index]
        
        a_sent_indices = sampled_sent_indices[index]
        a_sent_indices = np.where(a_sent_indices >= sent_num, -1, a_sent_indices)
        a_sent_indices = np.unique(a_sent_indices)
        sent_indices = []
        for a in a_sent_indices:
            if len(sent_indices)>FLAGS.sample_sent_num:break
            if a>-1:sent_indices.append(a)
        sent_indices = np.array(sent_indices)
        sent_indices.sort()
        all_sent_indices_sample.append(sent_indices)
        #print(sampled_sent_indices[index])
              
        sents = batch.sent_str_batch[index]
        summary_ext = []
        for sent_index in sent_indices:
            summary_ext.append(sents[sent_index])
        summary_ext = " ".join(summary_ext)
        all_summary_ext_sample.append(summary_ext)
              
        #entity_indices = np.unique(sampled_entity_indices[index])[:FLAGS.sample_entity_num]
        entity_num = batch.entity_nums[index]
        
        a_entity_indices = sampled_entity_indices[index]
        a_entity_indices = np.where(a_entity_indices >= entity_num, -1, a_entity_indices)
        a_entity_indices = np.unique(a_entity_indices)
        entity_indices = []
        for a in a_entity_indices:
            if len(entity_indices)==FLAGS.sample_entity_num:break
            if a>-1:entity_indices.append(a)
        entity_indices = np.array(entity_indices)
        entity_indices.sort()
        all_entity_indices_sample.append(entity_indices)
        
        #greedy
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
        #entity_indices = np.array(entity_indices)
        all_entity_indices_greedy.append(entity_indices)
        
        max_summary_ext_sample_len = max([len(a) for a in all_summary_ext_sample])
        max_summary_ext_greedy_len = max([len(a) for a in all_summary_ext_greedy])
        max_summary_ext_len = max_summary_ext_sample_len if max_summary_ext_sample_len > max_summary_ext_greedy_len else max_summary_ext_greedy_len
              
        #index, reps, summary_ext, max_enc_steps, entity_indices
    gen_batch_sample = batch.tile_for_greedy(all_summary_ext_sample, FLAGS.max_enc_steps, all_entity_indices_sample, max_summary_ext_len) 
    gen_batch_greedy = batch.tile_for_greedy(all_summary_ext_greedy, FLAGS.max_enc_steps, all_entity_indices_greedy, max_summary_ext_len)
    #print(max([len(a) for a in all_entity_indices_sample]), max([len(a) for a in all_entity_indices_greedy]))
    gen_batch = gen_batch_sample.concat(gen_batch_greedy)
    del gen_batch_sample, gen_batch_greedy 
    
    gen_outputs = gen_decoder.decode(gen_batch)
    gen_outputs_sample = gen_outputs[:FLAGS.batch_size]
    gen_outputs_greedy = gen_outputs[FLAGS.batch_size:2*FLAGS.batch_size]
              
    sample_rewards = []
    greedy_rewards = []
    for index in range(FLAGS.batch_size):
        #gen_rewards.append(rouge.get_scores(gen_outputs[index], batch.summary_str_batch[index] if batch.summary_str_batch[index]!="" else " ")[0]['rouge-2']['f']) 
        #print("t",gen_outputs[index]) 
        #print("s",batch.summary_str_batch[index])
        ref_summary = batch.summary_str_batch[index]
        ref_summary = ref_summary.split()
        #ref_summary = ref_summary if len(ref_summary) < FLAGS.max_dec_steps else ref_summary[:FLAGS.max_dec_steps]
        ref_summary = " ".join(ref_summary)
        sample_rewards.append(100 * rouge.get_scores(gen_outputs_sample[index],  ref_summary)[0]['rouge-1']['f'])  
        greedy_rewards.append(100 * rouge.get_scores(gen_outputs_greedy[index],  ref_summary)[0]['rouge-1']['f'])  
        
        for i in all_sent_indices_sample[index]: batch.rl_sent_label_batch[index, i] = 1./FLAGS.sample_sent_num 
        for i in all_entity_indices_sample[index]: batch.rl_entity_label_batch[index, i] = 1./FLAGS.sample_entity_num 
    #print("sample_rewards", sample_rewards)
    #print("greedy_rewards", greedy_rewards)
    batch.rl_reward_batch = np.array(sample_rewards) - np.array(greedy_rewards) 

def setup_training(model, batcher, gen_decoder=None):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph
  if FLAGS.convert_to_coverage_model:
    assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
    convert_to_coverage_model()
  if FLAGS.restore_best_model:
    restore_best_model()
  saver = tf.train.Saver(max_to_keep=FLAGS.ckpt_max_to_keep) # keep 50 checkpoints at a time

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=FLAGS.ckpt_save_model_secs, # save summaries for tensorboard every 60 secs
                     save_model_secs=FLAGS.ckpt_save_model_secs, # checkpoint every 60 secs
                     global_step=model.global_step)
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")
  try:
    run_training(model, batcher, sess_context_manager, sv, summary_writer, gen_decoder) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()
        
def training_greedy(model,batch,sess,gen_decoder):
    
    def entity_reward(summary, ref_entities):
        cover = 0
        for mentions in ref_entities:
            for mention in mentions:
                if summary.find(mention)>-1:
                    cover+=1
                    break
        return cover/len(ref_entities)
    
    batch.rl_sent_label_batch = np.zeros(np.shape(batch.sent_label_batch), dtype=np.float32)
    batch.rl_entity_label_batch = np.zeros(np.shape(batch.entity_label_batch), dtype=np.float32)
          
    #sampling
    samples = model.sampler(sess, batch)
    sampled_sent_indices = samples["sampled_sents"]
    sampled_entity_indices = samples["sampled_entities"]
    
    sent_scores = samples["sent_scores"]
    entity_scores = samples["entity_scores"]
    
    all_sent_indices_sample = []
    all_summary_ext_sample = []
    all_entity_indices_sample = []
    
    all_summary_ext_greedy = []
    all_entity_indices_greedy = []
    for index in range(FLAGS.batch_size):
        #sampler
        sent_num = batch.sent_nums[index]
        
        a_sent_indices = sampled_sent_indices[index]
        a_sent_indices = np.where(a_sent_indices >= sent_num, -1, a_sent_indices)
        a_sent_indices = np.unique(a_sent_indices)
        sent_indices = []
        for a in a_sent_indices:
            if len(sent_indices)>FLAGS.sample_sent_num:break
            if a>-1:sent_indices.append(a)
        sent_indices = np.array(sent_indices)
        sent_indices.sort()
        all_sent_indices_sample.append(sent_indices)
        #print(sampled_sent_indices[index])
              
        sents = batch.sent_str_batch[index]
        summary_ext = []
        for sent_index in sent_indices:
            summary_ext.append(sents[sent_index])
        summary_ext = " ".join(summary_ext)
        all_summary_ext_sample.append(summary_ext)
              
        #entity_indices = np.unique(sampled_entity_indices[index])[:FLAGS.sample_entity_num]
        entity_num = batch.entity_nums[index]
        
        a_entity_indices = sampled_entity_indices[index]
        a_entity_indices = np.where(a_entity_indices >= entity_num, -1, a_entity_indices)
        a_entity_indices = np.unique(a_entity_indices)
        entity_indices = []
        for a in a_entity_indices:
            if len(entity_indices)==FLAGS.sample_entity_num:break
            if a>-1:entity_indices.append(a)
        entity_indices = np.array(entity_indices)
        entity_indices.sort()
        all_entity_indices_sample.append(entity_indices)
        
        #greedy
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
              
        #index, reps, summary_ext, max_enc_steps, entity_indices
    gen_batch_sample = batch.tile_for_greedy(all_summary_ext_sample, FLAGS.max_enc_steps, all_entity_indices_sample)     
    gen_outputs_sample = gen_decoder.decode(gen_batch_sample)
    
    #gen_batch_greedy = batch.tile_for_greedy(all_summary_ext_greedy, FLAGS.max_enc_steps, all_entity_indices_greedy)     
    #gen_outputs_greedy = gen_decoder.decode(gen_batch_greedy)
    
    #sample_rewards = []
    #greedy_rewards = []
    rewards = []
    for index in range(FLAGS.batch_size):
        #gen_rewards.append(rouge.get_scores(gen_outputs[index], batch.summary_str_batch[index] if batch.summary_str_batch[index]!="" else " ")[0]['rouge-2']['f']) 
        #print("t",gen_outputs[index]) 
        #print("s",batch.summary_str_batch[index])
        ref_summary = batch.summary_str_batch[index]
        ref_summary = ref_summary.split()
        #ref_summary = ref_summary if len(ref_summary) < FLAGS.max_dec_steps else ref_summary[:FLAGS.max_dec_steps]
        ref_summary = " ".join(ref_summary)
        
        ref_entities = batch.mention_str_batch[index]
        #sample_rewards.append(100 * rouge.get_scores(gen_outputs_sample[index],  ref_summary)[0]['rouge-1']['f'])  
        #greedy_rewards.append(100 * rouge.get_scores(gen_outputs_greedy[index],  ref_summary)[0]['rouge-1']['f'])  
        try:
            sample_reward1 = rouge.get_scores(gen_outputs_sample[index],  ref_summary)[0]['rouge-1']['f']
            #sample_reward2 = entity_reward(gen_outputs_sample[index], ref_entities)
            #sample_reward = (1-FLAGS.entity_lambda) * sample_reward1 + FLAGS.entity_lambda * sample_reward2
            sample_reward = sample_reward1
            
            #greedy_reward1 = rouge.get_scores(gen_outputs_greedy[index],  ref_summary)[0]['rouge-1']['f']
            #greedy_reward2 = entity_reward(gen_outputs_greedy[index], ref_entities)
            #greedy_reward = (1-FLAGS.entity_lambda) * greedy_reward1 + FLAGS.entity_lambda * greedy_reward2
            #rewards.append(sample_reward - greedy_reward)
            rewards.append(sample_reward)
        except RuntimeError as e:
            ref_summary=" ".join(ref_summary.split(" ")[:FLAGS.max_dec_steps])
            sample_summary=gen_outputs_sample[index]
            sample_summary=" ".join(sample_summary.split(" ")[:FLAGS.max_dec_steps])
            #greedy_summary=gen_outputs_greedy[index]
            #greedy_summary=" ".join(greedy_summary.split(" ")[:FLAGS.max_dec_steps])
            sample_reward1 = rouge.get_scores(sample_summary,  ref_summary)[0]['rouge-1']['f']
            #sample_reward2 = entity_reward(gen_outputs_sample[index], ref_entities)
            #sample_reward = (1-FLAGS.entity_lambda) * sample_reward1 + FLAGS.entity_lambda * sample_reward2
            sample_reward = sample_reward1
            
            #greedy_reward1 = rouge.get_scores(gen_outputs_greedy[index],  ref_summary)[0]['rouge-1']['f']
            #greedy_reward2 = entity_reward(gen_outputs_greedy[index], ref_entities)
            #greedy_reward = (1-FLAGS.entity_lambda) * greedy_reward1 + FLAGS.entity_lambda * greedy_reward2
            
            #greedy_reward = rouge.get_scores(greedy_summary,  ref_summary)[0]['rouge-1']['f']
            #rewards.append(sample_reward - greedy_reward)
            rewards.append(sample_reward)
            print(gen_outputs_sample[index])
            #print(gen_outputs_greedy[index])
            print(ref_summary)
        
        for i in all_sent_indices_sample[index]: batch.rl_sent_label_batch[index, i] = 1./FLAGS.sample_sent_num 
        for i in all_entity_indices_sample[index]: batch.rl_entity_label_batch[index, i] = 1./FLAGS.sample_entity_num 
    
    #batch.rl_reward_batch = np.array(sample_rewards) - np.array(greedy_rewards)
    batch.rl_reward_batch = np.array(rewards) 

def run_training(model, batcher, sess_context_manager, sv, summary_writer, gen_decoder=None):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  with sess_context_manager as sess:
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
    t0 = time.time()
    losses = 0.
    while True: # repeats until interrupted
      batch = batcher.next_batch()
      
      if FLAGS.rl_lambda > 0:
          #gen_summary
          
          training_greedy(model,batch, sess, gen_decoder)
          
          #print(batch.rl_reward_batch)
          #print(batch.rl_sent_label_batch)
          #print(batch.rl_entity_label_batch)
      
      #FLAGS.batch_size = batch_size
      
      #tf.logging.info('running training step %d' % train_step)
      results = model.run_train_step(sess, batch)
      t1=time.time()
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      train_step = results['global_step'] # we need this to update our running average loss
      loss = results['loss']
      #scores = results["scores"][0]
      #labels = results["labels"][0]
      
      #with open("out7.txt", "w") as w_file:
          #print >>w_file, results["vocab_scores"]
          #for iii in results["vocab_scores"]:
            #print >>w_file, " ".join([str(i) for i in iii])
      #print(results["data_name"])
      if not np.isfinite(loss):
        #print(results["data_name"])
        raise Exception("Loss is not finite. Stopping.")
    
      #if FLAGS.coverage:
      #  coverage_loss = results['coverage_loss']
      #  coverage_losses += coverage_loss
      
      summary_writer.add_summary(summaries, train_step) # write the summaries
      
      losses += loss
      
      if train_step % 100 == 0:
          t1 = time.time()
          tf.logging.info('seconds for training step %d: %.3f' % (train_step, t1-t0))
          tf.logging.info('losses: %f', losses / 100.) # print the loss to screen
          #tf.logging.info('scores:%s' % " ".join([str(item) for item in scores]))
          #tf.logging.info('labels:%s' % " ".join([str(item) for item in labels]))
          losses = 0.
          t0 = t1

          #if FLAGS.coverage:
              #coverage_loss = results['coverage_loss']
              #tf.logging.info("coverage_losses: %f", coverage_losses / 100.) # print the coverage loss to screen
              #coverage_losses = 0.
      
          summary_writer.flush()


def run_eval(model, batcher, vocab):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far

  while True:
    _ = util.load_ckpt(saver, sess) # load a new checkpoint
    batch = batcher.next_batch() # get the next batch

    # run eval on the batch
    t0=time.time()
    results = model.run_eval_step(sess, batch)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)

    # print the loss and coverage loss to screen
    loss = results['loss']
    tf.logging.info('loss: %f', loss)
    if FLAGS.coverage:
      coverage_loss = results['coverage_loss']
      tf.logging.info("coverage_loss: %f", coverage_loss)

    # add summaries
    summaries = results['summaries']
    train_step = results['global_step']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()


def test(batcher):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        while True:
            try:
                batch = batcher.next_batch()
                entities2sents = tf.placeholder(tf.float32, [1, None, None], name='entities2sents')
                feed_dict = {}
                feed_dict[entities2sents] = batch.entities2sents
                entity_es_D_left=tf.matrix_inverse(tf.matrix_diag(tf.reduce_sum(entities2sents, 2) + tf.constant(1, tf.float32)) ** 0.5)
                entity_es_D_right = tf.matrix_inverse(tf.matrix_diag(tf.reduce_sum(entities2sents, 1) + tf.constant(1, tf.float32)) ** 0.5)
                m1 = tf.matmul(entity_es_D_left, entities2sents)
                
                m2 = tf.matmul(m1, entity_es_D_right)
                
                m3 = tf.matmul(entities2sents, entity_es_D_right)
            
                
                sess.run(m2, feed_dict)
                del entity_es_D_left, entity_es_D_right, m1, m2, m3
                #print(sess.run(new_entity_es_enc, feed_dict))
            except:
                print("error")
                

def test2(batcher):
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        while True:
            #try:
                batch = batcher.next_batch()
                entities2sents = np.squeeze(batch.entities2sents,0)
                #print(entities2sents)
                #print(np.sum(entities2sents, axis=1))
                #print(np.diag(np.sum(entities2sents, axis=1)))
                entity_es_D_left=np.linalg.pinv(np.diag(np.sum(entities2sents, 1) + 1e-3) ** 0.5)
                entity_es_D_right = np.linalg.pinv(np.diag(np.sum(entities2sents, 0) + 1e-3) ** 0.5)
                m1 = np.matmul(entity_es_D_left, entities2sents)
                
                m2 = np.matmul(m1, entity_es_D_right)
                
                m3 = tf.matmul(entities2sents, entity_es_D_right)
                print(m2)
                del entity_es_D_left, entity_es_D_right, m1, m2, m3
                
                entities2sents = tf.placeholder(tf.float32, [1, None, None], name='entities2sents')
                feed_dict = {}
                feed_dict[entities2sents] = batch.entities2sents
                entity_es_D_left=tf.matrix_inverse(tf.matrix_diag(tf.reduce_sum(entities2sents, 2) + tf.constant(1e-3, tf.float32)) ** 0.5)
                entity_es_D_right = tf.matrix_inverse(tf.matrix_diag(tf.reduce_sum(entities2sents, 1) + tf.constant(1e-3, tf.float32)) ** 0.5)
                m1 = tf.matmul(entity_es_D_left, entities2sents)
                
                m2 = tf.matmul(m1, entity_es_D_right)
                
                m3 = tf.matmul(entities2sents, entity_es_D_right)
            
                
                print(sess.run(m2, feed_dict))
                del entity_es_D_left, entity_es_D_right, m1, m2, m3
                
                break
                #print(sess.run(new_entity_es_enc, feed_dict))
            #except:
            #    print("error")
        

def main(unused_argv):

  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  word_vocab = Vocab(FLAGS.word_vocab_path, FLAGS.word_vocab_size) # create a word vocabulary
  entity_vocab = Vocab(FLAGS.entity_vocab_path, FLAGS.entity_vocab_size) # create a entity vocabulary

  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
  #if FLAGS.mode == 'decode':
  #  FLAGS.batch_size = FLAGS.beam_size

  # If single_pass=True, check we're in decode mode
  if FLAGS.single_pass and (FLAGS.mode!='decode' and FLAGS.mode!='ext_greedy'):
    raise Exception("The single_pass flag should only be True in decode mode")

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 
                 'max_grad_norm', 'encoder_hidden_dim', 'hidden_dim', 'word_emb_dim', 'entity_emb_dim', 
                 'batch_size', 'max_dec_steps', 'max_enc_steps', "min_dec_steps", 
                 "use_entity", "use_mention_occurs", "use_entity_embedding", "corpus", "use_gcn_entity", 
                 "max_position", "gcn_level", "pos_emb_type", "max_sent_num", "max_entity_num", 
                 "entity_lambda", "rl_lambda", "pointer_gen", "cov_loss_wt", 
                 "gen_log_root", "gen_exp_name", "coverage", "beam_size", "entity_vocab_size", 
                 "sample_sent_num", "sample_entity_num", "dec_method", "write_mode", "lambda_ee_train"]
  hps_dict = {}
  gen_hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
      gen_hps_dict[key] = val
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
  gen_hps = namedtuple("HParams", gen_hps_dict.keys())(**gen_hps_dict)
  #gen_hps.batch_size = FLAGS.beam_size
  gen_hps = gen_hps._replace( mode="decode", dec_method="greedy", use_entity_embedding=FLAGS.gen_use_entity_embedding)

  # Create a batcher object that will create minibatches of data
  batcher = Batcher(FLAGS.data_path, word_vocab, entity_vocab, hps, single_pass=FLAGS.single_pass)

  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'train':
    print ("creating model...")
    model = SummarizationModel_GCN(hps, word_vocab, entity_vocab)
    gen_decoder = None
    gen_decoder = gen_model(word_vocab, entity_vocab, gen_hps)
    model.frozen_vars = gen_decoder.frozen_vars
    setup_training(model, batcher, gen_decoder)
  elif hps.mode == 'eval':
    model = SummarizationModel_GCN(hps, word_vocab, entity_vocab)
    run_eval(model, batcher, word_vocab, entity_vocab)
  elif hps.mode == 'decode':
    ext_hps = hps#._replace(batch_size=1)
    model_ext = SummarizationModel_GCN(hps, word_vocab, entity_vocab)
    
    FLAGS.gen_log_root = os.path.join(FLAGS.gen_log_root, FLAGS.gen_exp_name)
    tf.set_random_seed(111) # a seed value for randomness
    gen_model_hps = hps
    gen_model_hps = hps._replace(max_dec_steps=1)
    if FLAGS.write_mode=="decode":
        gen_model_hps = gen_model_hps._replace(batch_size=FLAGS.beam_size, dec_method="beam-search", use_entity_embedding=FLAGS.gen_use_entity_embedding)
    else:
        gen_model_hps = gen_model_hps._replace(dec_method="greedy", use_entity_embedding=FLAGS.gen_use_entity_embedding)
    
    model_gen=None
    if FLAGS.decode_gen:
        model_gen = SummarizationModel(gen_model_hps, word_vocab, entity_vocab)
    
    decoder = ExtGenDecoder(model_ext, model_gen, batcher, word_vocab, ext_hps.write_mode)
    decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
  elif hps.mode == "ext_greedy":
    decode_model_hps = hps  # This will be the hyperparameters for the decoder model
    decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
    model = SummarizationModel_GCN(decode_model_hps, word_vocab, entity_vocab)
    decoder = ExtGreedyDecoder(model_ext, model_gen, batcher)
    decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
  elif hps.mode == "test":
    print("test")
    test2(batcher)
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")



if __name__ == '__main__':
  tf.app.run()











