import re
import hashlib
import rouge

path_base = "/home/test/kbsumm"
path_corpus = path_base + "/data/cnn+dm"
path_parser=r"/home/test/eclipse/libs/stanford-parser/stanford-parser.jar"

_ROUGE_PATH = "/home/test/pyrouge-master/tools/ROUGE-1.5.5"
_PYROUGE_TEMP_FILE = path_corpus+"/tmp"

path_tokens=path_corpus+"/tokens"
path_stories=path_corpus+"/stories"
path_stories_t=path_corpus+"/stories-t"
path_stories_ts=path_corpus+"/stories-ts"
path_stories_tsa=path_corpus+"/stories-tsa"

path_mentions=path_corpus+"/mentions"
path_mentions_ex=path_corpus+"/mentions-ex"

path_entities=path_corpus+"/entities"
path_entities2mentions=path_corpus+"/entities2mentions"
path_entities2mentions_labeled=path_corpus+"/entities2mentions-labeled"

path_texts_ts=path_corpus+"/texts-ts"
path_texts_tsl=path_corpus+"/texts-tsl"
path_texts_tslr=path_corpus+"/texts-tslr"
path_texts_tsla=path_corpus+"/texts-tsla"

path_summaries_ts=path_corpus+"/summaries-ts"
path_summaries_tsl=path_corpus+"/summaries-tsl"
path_summaries_tslr=path_corpus+"/summaries-tslr"
path_summaries_tsla=path_corpus+"/summaries-tsla"

path_summaries_ext_tsl=path_corpus+"/summaries-ext-tsl"
path_summaries_label_tsl=path_corpus+"/summaries-label-tsl"

path_summaries_ext_tsl_2=path_corpus+"/summaries-ext-tsl-2"
path_summaries_label_tsl_2=path_corpus+"/summaries-label-tsl-2"

path_kg=path_base+"/data/yagoFacts.ttl"
path_entity_vocab_all=path_corpus+"/entity-vocab-all"
path_entity_embeddings_all=path_corpus+"/entity-embeddings-all"
path_entity_vocab_4=path_corpus+"/entity-vocab-40000"
path_entity_vocab_embeddings_4=path_corpus+"/entity-embeddings-40000-128"
path_entity_vocab_5=path_corpus+"/entity-vocab-50000"
path_entity_vocab_embeddings_5=path_corpus+"/entity-embeddings-50000-128"
path_entity_vocab_10=path_corpus+"/entity-vocab-100000"
path_entity_vocab_embeddings_10=path_corpus+"/entity-embeddings-100000-128"
path_entity_vocab_20=path_corpus+"/entity-vocab-200000"
path_entity_vocab_embeddings_20=path_corpus+"/entity-embeddings-200000-128"
path_entity_vocab_30=path_corpus+"/entity-vocab-300000"
path_entity_vocab_embeddings_30=path_corpus+"/entity-embeddings-300000-128"
path_entity_vocab_50=path_corpus+"/entity-vocab-500000"
path_entity_vocab_embeddings_50=path_corpus+"/entity-embeddings-500000-128"
path_entity_vocab=path_entity_vocab_20
path_entity_vocab_embeddings=path_entity_vocab_embeddings_50

path_word_embeddings=path_base+"/data/glove/wiki_dm_cnn_vectors.txt"
path_word_vocab=path_corpus+"/word-vocab-40000"
path_word_vocab_embeddings_word2vec=path_corpus+'/word-vocab-embeddings-word2vec-40000-128'
path_word_vocab=path_corpus+"/word-vocab-40000"
path_word_vocab_embeddings_word2vec=path_corpus+'/word-vocab-embeddings-word2vec-40000-128'
path_word_vocab_embeddings=path_word_vocab_embeddings_word2vec
#0e247a034a
path_train_file=path_corpus+"/trains"
path_dev_file=path_corpus+"/devs"
path_test_file=path_corpus+"/tests"

path_test_test_file=path_corpus+"/test-test-test"

path_code=path_base + '/codes/code1'

path_chunked=path_corpus + "/chunked-gcn"
path_chunked_train=path_chunked + "/train"
path_chunked_dev=path_chunked + "/dev"
path_chunked_test=path_chunked + "/test"

path_chunked_gcn=path_corpus + '/chunked-gcn'
path_chunked_train_gcn=path_chunked_gcn + "/train"
path_chunked_dev_gcn=path_chunked_gcn + "/dev"
path_chunked_test_gcn=path_chunked_gcn + "/test"

path_log_root = path_corpus + "/models"
path_log_root_gcn = path_corpus + "/models-gcn-1010"

def Hashhex(s):
  h = hashlib.sha1()
  h.update(s.encode("utf8"))
  return h.hexdigest()

def preprocess_line(tokenizer,line=''):
    line=line.lower()
    line=' '.join(tokenizer.tokenize(line)).replace('`','\'')
    
    line=re.sub(r"-LRB-","(", line, 0)
    line=re.sub(r"-RRB-",")", line, 0)
    line=re.sub(r"''","\"", line, 0)
    line=re.sub(r"``","\"", line, 0)
    line=re.sub(r'[^\x00-\x7F]','', line, 0)
    line=re.sub(r'([0-9]+(,[0-9]{3})*)(.[0-9]{1,2})?','[NUM]', line, 0)
    return line

def preprocess_line2(line=''):
    #line=line.lower()
    line=re.sub(r"''","\"", line, 0)
    line=re.sub(r"``","\"", line, 0)
    line=re.sub(r'[^\x00-\x7F]', '', line, 0)
    line=re.sub(r'([0-9]+(,[0-9]{3})*)(.[0-9]{1,2})?','[NUM]', line, 0)
    return line

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=4,
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)
def evaluate(ref, hyp):
    all_hypothesis = [hyp]
    all_references = [ref]
    scores = evaluator.get_scores(all_hypothesis, all_references)
    return scores

#print(evaluate("my name is cjq", "cjq"))



