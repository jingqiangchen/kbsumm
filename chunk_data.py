# -*-coding:utf-8 -*
import sys
import re
# 设置系统默认编码，执行dir（sys）时不会看到这个方法，在解释器中执行不通过，可以先执行reload(sys)，
#在执行 setdefaultencoding('utf-8')，此时将系统默认编码设置为utf-8。（见设置系统默认编码 ）
reload(sys)
sys.setdefaultencoding('utf-8')#添加该方法声明编码-

import os
import hashlib
import struct
import numpy as np
from tensorflow.core.example import example_pb2

import common
from data import Vocab
import math

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

path_train_file = common.path_train_file #common.path_test_test_file
path_dev_file = common.path_dev_file 
path_test_file = common.path_test_file
path_example = common.path_corpus+"/example"
#0e247a034a']
path_chunks = common.path_chunked
path_chunks_gcn = common.path_chunked_gcn 

path_summaries_label_tsl=common.path_summaries_label_tsl_2
path_summaries_ext=common.path_summaries_ext_tsl_2
path_summaries=common.path_summaries_tsl
path_texts=common.path_texts_tsl

path_entity_vocab=common.path_entity_vocab
path_entity_embeddings=common.path_entity_vocab_embeddings

path_entities=common.path_entities
path_mentions_ex=common.path_mentions_ex
path_entities2mentions=common.path_entities2mentions
path_entities2mentions_labeled=common.path_entities2mentions_labeled
path_entities=common.path_entities

path_entity2id=common.path_base+"/codes/aidalight/data/bned/resources/entity_id"
path_entity2entity=common.path_base+"/codes/aidalight/data/bned/resources/entity_entity_counter"
path_entityCount=common.path_base+"/codes/aidalight/data/bned/resources/entity_counter"

NUMBER_ENTITIES = 2638982
WORD_VOCAB_SIZE = 40004
ENTITY_VOCAB_SIZE = 40001
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def chunk_file(path_chunks, set_name):
  in_file = path_chunks + '/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  if not os.path.isdir(path_chunks+"/"+set_name):
    os.mkdir(path_chunks+"/"+set_name)
  
  while not finished:
    chunk_fname = os.path.join(path_chunks+"/"+set_name, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(path_chunks):
  # Make a dir to hold the chunks
  if not os.path.isdir(path_chunks):
    os.mkdir(path_chunks)
  # Chunk the data
  for set_name in ['train', 'dev', 'test']:
  #for set_name in ['example-1']:
    print ("Splitting %s data into chunks..." % set_name)
    chunk_file(path_chunks, set_name)
  print ("Saved chunked data in %s" % path_chunks)


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def write_to_bin(path_in_file, path_out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print ("Making bin file for URLs listed in %s..." % path_in_file)
  file_list = read_text_file(path_in_file)
  num_stories = len(file_list)

  with open(path_out_file, 'wb') as writer:
    for idx,s in enumerate(file_list):
      if idx % 1000 == 0:
        print ("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      elif os.path.isfile(os.path.join(path_summaries_ext, s)):
        story_file = os.path.join(path_summaries_ext, s)
      else:
        print ("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, path_summaries_ext))
        # Check again if tokenized stories directories contain correct number of files

      # Get the strings to write to .bin file
      #print(s)
      with open(path_summaries_ext + "/" + s) as file: 
          summary_ext = re.sub("\s+", " ", file.read().replace("\n", " "))
      with open(path_summaries + "/" + s) as file: 
          summary = re.sub("\s+", " ", file.read().replace("\n", " "))
          summary2 = "|||".join(summary)
      
      all_entities={}
      key2keys={}
      with open(path_entities + "/" + s) as file: 
          for line in file.readlines():
              line=line.strip()
              items=line.split("\t")
              all_entities[items[0]]=items[2]
              keys=items[0].strip("+").split("+")
              for key in keys:
                  key2keys[key]=items[0]
      
      #print(s)
      entities = []
      mentions = []
      entities2mentions = {}
      entity_ids=[]
      with open(path_entities2mentions_labeled + "/" + s) as file: 
          for line in file.readlines():
              line=line.strip()
              items=line.split("|||")
              entity_ids.append(items[0])
              entities2mentions[items[0]]=items
                  
      for entity_id in entity_ids:
          if entities2mentions[entity_id][-1]=="1":
              keys=key2keys[entity_id]
              entities.append(all_entities[keys])
              keys=keys.strip("+").split("+")
              ss=""
              for key in keys:
                if key in entities2mentions:
                    entities2mentions[entity_id][-1]="0"
                    ss+="|||".join(entities2mentions[key][1:-1])+"|||"
              mentions.append(ss.strip("|||"))
              
      entity_txt="|||".join(entities)
      
      mention_txt=""
      mention_count=""
      for entity, mention in zip(entities, mentions):
          ms=mention.split("|||")
          ms_unique=[]
          ms_count={}
          for m in ms:
              if m not in ms_count:
                  ms_count[m]=0
                  ms_unique.append(m)
              ms_count[m]+=1
          ms_count2=[]
          for m in ms_unique:
              ms_count2.append(str(ms_count[m]))
          mention_txt+="||".join(ms_unique)+"|||"
          mention_count+="||".join(ms_count2)+"|||"
      mention_txt=mention_txt.strip("|||")   
      mention_count=mention_count.strip("|||")   

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['data_name'].bytes_list.value.extend([s.encode('utf-8')])
      tf_example.features.feature['summary_ext'].bytes_list.value.extend([summary_ext.encode('utf-8')])
      tf_example.features.feature['summary'].bytes_list.value.extend([summary.encode('utf-8')])
      tf_example.features.feature['entities'].bytes_list.value.extend([entity_txt.encode('utf-8')])
      tf_example.features.feature['entities2mentions'].bytes_list.value.extend([mention_txt.encode('utf-8')])
      tf_example.features.feature['entities2mentions_count'].bytes_list.value.extend([mention_count.encode('utf-8')])
      
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

  print ("Finished writing file %s\n" % path_out_file)
  
  
def write_to_bin_gcn(path_in_file, path_out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print ("Making bin file for URLs listed in %s..." % path_in_file)
  file_list = read_text_file(path_in_file)
  num_stories = len(file_list)
  
  max_sent_num = 0
  all_entity2entity={}
  all_entity_count={}
  with open(path_entity2entity) as file:
    for line in file.readlines():
        line=line.strip()
        items=line.split("\t")
        all_entity2entity[int(items[0])]=int(items[1])
  with open(path_entityCount) as file:
    for line in file.readlines():
        line=line.strip()
        items=line.split("\t")
        all_entity_count[int(items[0])]=int(items[1])
  
        
  all_entity2id={}
  with open(path_entity2id) as file:
    for line in file.readlines():
        line=line.strip()
        items=line.split("\t")
        all_entity2id[items[0].decode('unicode_escape')]=int(items[1])

  with open(path_out_file, 'wb') as writer:
    for idx,s in enumerate(file_list):
      if idx % 1000 == 0:
        print ("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      elif os.path.isfile(os.path.join(path_summaries_ext, s)):
        story_file = os.path.join(path_summaries_ext, s)
      else:
        print(os.path.join(path_summaries_ext, s))
        print ("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, path_summaries_ext))
        # Check again if tokenized stories directories contain correct number of files

      with open(path_summaries_ext + "/" + s) as file: 
          summary_ext=""
          for line in file.readlines():
              line=line.strip("\n").strip(" ")
              if line=="":continue
              if line[-1]!='.' and line[-1]!='?' and line[-1]!='!' and line[-1]!=':': 
                  line=line+" . "
              else: line+=" "
              summary_ext += re.sub("\s+", " ", line)

      with open(path_summaries + "/" + s) as file: 
          summary=""
          summary2=""
          for line in file.readlines():
              line=line.strip("\n").strip(" ")
              if line=="": continue
              if line[-1]!='.' and line[-1]!='?' and line[-1]!='!' and line[-1]!=':': 
                  line=line+" . "
              else: line+=" "
              line=re.sub("\s+", " ", line)
              summary += line
              summary2 += line+"|||"
          summary2=summary2.strip("|||")
          #print(summary2)

      # Get the strings to write to .bin file
      #print(s)
      summary_label=np.fromfile(path_summaries_label_tsl + "/" + s, np.float32)
          
      with open(path_texts + "/" + s) as file:
          sents = []
          for sent in file.readlines():
              sent=sent.strip("\n").strip(" ")
              if sent!="": 
                  if line[-1]!='.' and line[-1]!='?' and line[-1]!='!' and line[-1]!=':': 
                      sent+=" . "
                  else: sent+=" "
                  sents.append(sent)
          
          num_sents = len(sents)
          if max_sent_num < num_sents: max_sent_num = num_sents

          sent_relas = np.zeros([num_sents, num_sents], np.int32)
          for i in range(1, num_sents-1):
              sent_relas[i, i-1] = sent_relas[i, i+1] = 1
          try:
              sent_relas[0, 1] = 1
              sent_relas[num_sents-1, num_sents-2] = 1
          except:
              print(s, num_sents)
          
      text="\n".join(sents)
      if len(summary_label)!=num_sents:
          print(s, len(summary_label), num_sents)
      
      mentions2sents={}
      with open(path_mentions_ex+"/"+s+".txt") as file:
          for line in file.readlines():
              items=line.strip().split("\t")
              if items[0] not in mentions2sents:
                  mentions2sents[items[0]]=[]
              sent_no=int(items[2])
              if sent_no<num_sents:
                  mentions2sents[items[0]].append(sent_no)  
      
      all_entities={}
      key2keys={}
      with open(path_entities + "/" + s) as file: 
          for line in file.readlines():
              line=line.strip()
              items=line.split("\t")
              all_entities[items[0]]=items[2]
              keys=items[0].strip("+").split("+")
              for key in keys:
                  key2keys[key]=items[0]
      
      mentions = []
      entities2mentions = {}
      entity_ids=[]
      with open(path_entities2mentions_labeled + "/" + s) as file: 
          for line in file.readlines():
              line=line.strip()
              items=line.split("|||")
              i=0
              while i<len(items):
                  if items[i].strip(" ")=="":
                      del items[i]
                  else: i+=1
              if len(items)<=2:continue
              entity_ids.append(items[0])
              entities2mentions[items[0]]=items
      
      entities = []
      entity_labels=[]
      entity_orders=[]
      for entity_id in entity_ids:
          if entity_id not in entity_orders:
              entity_labels.append(int(entities2mentions[entity_id][-1]))
              keys=key2keys[entity_id]
              entities.append(all_entities[keys])
              keys=keys.strip("+").split("+")
              ss=""
              for key in keys:
                  if key in entities2mentions:
                      entity_orders.append(key)
                      ss+="|||".join(entities2mentions[key][1:-1])+"|||"
              mentions.append(ss.strip("|||"))
      entity_labels=np.array(entity_labels, dtype=np.int32)        
      entity_txt="|||".join(entities)
      
      mention_txt=""
      mention_count=""
      for entity, mention in zip(entities, mentions):
          ms=mention.split("|||")
          ms_unique=[]
          ms_count={}
          for m in ms:
              if m not in ms_count:
                  ms_count[m]=0
                  ms_unique.append(m)
              ms_count[m]+=1
          ms_count2=[]
          for m in ms_unique:
              ms_count2.append(str(ms_count[m]))
          mention_txt+="||".join(ms_unique)+"|||"
          mention_count+="||".join(ms_count2)+"|||"
      mention_txt=mention_txt.strip("|||")   
      mention_count=mention_count.strip("|||")   
      
      num_entities=len(entities)
      entities2sents=np.zeros([num_entities, num_sents], np.int32)
      entity_orders=[]
      entity_index=0
      for entity_id in entity_ids:
          if entity_id not in entity_orders:
              keys=key2keys[entity_id]
              keys=keys.strip("+").split("+")
              ss=""
              for key in keys:
                  if key in entities2mentions:
                      entity_orders.append(key)
                      for sent_index in mentions2sents[key]:
                          if sent_index<num_sents:
                              entities2sents[entity_index, sent_index]+=1
              mentions.append(ss.strip("|||"))
              entity_index+=1
      
      MAX = 4000000
      entities2entities=np.zeros([num_entities, num_entities], np.int32)
      for i in range(num_entities):
          if entities[i] not in all_entity2id: continue
          entity_id1=all_entity2id[entities[i]]
          for j in range(num_entities):
              if entities[j] not in all_entity2id: continue
              entity_id2=all_entity2id[entities[j]]
              if entity_id1>entity_id2:
                  entity_id1, entity_id2 = entity_id2, entity_id1
              key=entity_id1*MAX+entity_id2
              if key not in all_entity2entity: continue
              entities2entities[i][j]=entity_coh(all_entity_count[entity_id1],all_entity_count[entity_id2],
                                                 all_entity2entity[entity_id1*MAX+entity_id2],entities[i],entities[j])
      
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['data_name'].bytes_list.value.extend([s.encode('utf-8')])
      tf_example.features.feature['summary_ext'].bytes_list.value.extend([summary_ext.encode('utf-8')])
      tf_example.features.feature['summary'].bytes_list.value.extend([summary.encode('utf-8')])
      tf_example.features.feature['summary2'].bytes_list.value.extend([summary2.encode('utf-8')])
      tf_example.features.feature['text'].bytes_list.value.extend([text.encode('utf-8')])
      tf_example.features.feature['sent_relas'].bytes_list.value.extend([sent_relas.tobytes()])
      tf_example.features.feature['entities2sents'].bytes_list.value.extend([entities2sents.tobytes()])
      tf_example.features.feature['entities2entities'].bytes_list.value.extend([entities2entities.tobytes()])
      tf_example.features.feature['summary_label'].bytes_list.value.extend([summary_label.tobytes()])
      tf_example.features.feature['entities'].bytes_list.value.extend([entity_txt.encode('utf-8')])
      tf_example.features.feature['entity_labels'].bytes_list.value.extend([entity_labels.tobytes()])
      tf_example.features.feature['entities2mentions'].bytes_list.value.extend([mention_txt.encode('utf-8')])
      tf_example.features.feature['entities2mentions_count'].bytes_list.value.extend([mention_count.encode('utf-8')])
      
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

  print ("Finished writing file %s\n" % path_out_file)
  print (max_sent_num)

def entity_coh(e1,e2,ee,a=None,b=None):
    if ee==0 or e1==0 or e2==0: return 0
    else: 
        #print(a,b,NUMBER_ENTITIES,e1,e2,ee)
        #print( math.log(max([e1,e2]),2))
        #print(math.log(ee,2))
        #print(math.log(NUMBER_ENTITIES,2))
        #print(math.log(min([e1,e2]),2))
        ss=1-(math.log(max([e1,e2]),2)-math.log(ee,2))/(math.log(NUMBER_ENTITIES,2)-math.log(min([e1,e2]),2))
        print(a,b,ss)
        return ss

def build_word2vec_embeddings():
    import word2vec
    def build_total_embeddings():
        word2vec.word2vec('/home/test/kbsumm/data/glove/wiki-dm-cnn.txt', '/home/test/kbsumm/data/glove/wiki-dm-cnn-word2vec-128.bin', size=128, verbose=True)
        
    def vocab_embedding():
        model = word2vec.load('/home/test/kbsumm/data/glove/wiki-dm-cnn-word2vec-128.bin')
        with open(common.path_word_vocab) as file:
            vocab=[line.split('\t')[0] for line in file.readlines()]
            embeddings=[]
            for word in vocab:
                embeddings.append(model[word])
            embeddings=np.array(embeddings,np.float32)
            embeddings.tofile(common.path_word_vocab_embeddings_word2vec)
    
    #build_total_embeddings()
    vocab_embedding()
    

def chunk_for_gen():
    write_to_bin(path_train_file, os.path.join(path_chunks, "train.bin"))
    write_to_bin(path_dev_file, os.path.join(path_chunks, "dev.bin"))
    write_to_bin(path_test_file, os.path.join(path_chunks, "test.bin"))

    chunk_all(path_chunks)


def chunk_for_gcn():
    write_to_bin_gcn(path_train_file, os.path.join(path_chunks_gcn, "train.bin"))
    write_to_bin_gcn(path_dev_file, os.path.join(path_chunks_gcn, "dev.bin"))
    write_to_bin_gcn(path_test_file, os.path.join(path_chunks_gcn, "test.bin"))

    chunk_all(path_chunks_gcn)

if __name__ == '__main__':
  #copy_files()
  
  #chunk_for_gen()
  
  chunk_for_gcn()
  #test3(path_train_file, os.path.join(path_chunks_gcn, "train.bin"))
  
  #build_word2vec_embeddings()
  
  #test()
  #test2(path_test_file, os.path.join(path_chunks_gcn, "train.bin"))
































