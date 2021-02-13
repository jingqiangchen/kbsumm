import common
from readline import read_history_file
import rdflib
import pandas as pd
from converters import rdflib_to_kg
from rdf2vec import RDF2VecTransformer
from multiprocessing.pool import ThreadPool, Pool
#from itertools import izip
from itertools import repeat

import numpy as np
import os
import collections

from walkers import (RandomWalker, WeisfeilerLehmanWalker, 
                     AnonymousWalker, WalkletWalker, NGramWalker,
                     CommunityWalker, HalkWalker)
from pandas._libs.tslibs.offsets import key
from numpy import source

def LoadTokenMapping(filename):

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for token_mapping in line.split(';'):
      if not token_mapping:
        continue

      start, length = token_mapping.split(',')

      mapping.append((int(start), int(start) + int(length)))

    mapping.sort(key=lambda x: x[1])  # Sort by start.

  return mapping


def LoadEntityMapping(filename):

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for entity_mapping in line.split(';'):
      if not entity_mapping:
        continue

      entity_index, start, end = entity_mapping.split(',')

      mapping.append((int(entity_index), int(start), int(end)))

    mapping.sort(key=lambda x: x[1]-x[2])  # Sort by start.

  return mapping


def create_train_dev_test():
    def create():
        file_names={"trains":"wayback_training_urls.txt", "devs":"wayback_validation_urls.txt", "tests":"wayback_test_urls.txt"}
        for key in file_names.keys():
            file=open(common.path_corpus+"/"+file_names[key])
            w_file=open(common.path_corpus+"/"+key, "w")
            for line in file.readlines():
                line=line.strip()
                print(line)
                w_file.write(common.Hashhex(line)+"\n")
            w_file.close()
            file.close()
            
    def removeError():
        files={common.path_train_file, common.path_dev_file, common.path_test_file}
        error_file=open(common.path_corpus+"/train-error")
        errors=[line.strip() for line in error_file.readlines()]
        error_file.close()
        for key in files:
            lines=[]
            file=open(key)
            for line in file.readlines():
                line=line.strip()
                if line not in errors:
                    lines.append(line)
            file.close()
            
            w_file=open(key,"w")
            for line in lines:
                w_file.write(line+"\n")
            w_file.close()
            
    #create()
    removeError()


def story_preprocess():
    file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
    #file_names=[common.path_test_test_file]
    def tokenize():
        count=0
        error_file=open(common.path_train_file+"-error","w")
        for file_name in file_names:
            file=open(file_name)
            for line in file.readlines():
                line=line.strip()
                token_mapping=LoadTokenMapping(common.path_tokens+"/"+line+".txt")
                story_file=open(common.path_stories+"/"+line+".story")
                story_s_file=open(common.path_stories_t+"/"+line+".story","w") 
                text=story_file.read().encode("utf8")
                #print(text.decode("utf8"))
                sign=0
                len_text=len(text)
                
                try:
                    pre_end=0
                    for (start, end) in token_mapping:
                        
                        for i in range(pre_end,start):
                            #print(chr(text[i]))
                            if chr(text[i])=='\n':
                                #print(text[i])
                                story_s_file.write("\n")
                                sign=0
                        if sign>0:
                            story_s_file.write(" ")
                        #print(text[start:end+1])
                        story_s_file.write(text[start:end+1].decode("utf8"))
                        sign+=1
                        pre_end=end
                        #print(start, end)
                        
                    story_s_file.close()
                    story_file.close()
                    count+=1
                    print("tokenize: %d-%s" % (count, line))
                except:
                    error_file.write(line+"\n")
            error_file.close()
            file.close()

    def tokenize2():
        count=0
        error_file=open(common.path_train_file+"-error","w")
        for file_name in file_names:
            file=open(file_name)
            for story_file_name in file.readlines():
                story_file_name=story_file_name.strip()
                token_mapping=LoadTokenMapping(common.path_tokens+"/"+story_file_name+".txt")
                story_file=open(common.path_stories+"/"+story_file_name+".story")
                text=story_file.read().encode("utf8")
                story_file.close()
                
                story_file=open(common.path_stories+"/"+story_file_name+".story")                
                
                story_s_file=open(common.path_stories_t+"/"+story_file_name+".story","w") 
                
                #print(text.decode("utf8"))
                mapping_len=len(token_mapping)
                index=0
                try:
                    for line in story_file.readlines():
                        sign=0
                        line=line.replace(" ","").replace("\n","").replace("\t","").replace(chr(0xa0), "")
                        if line=="":
                            story_s_file.write("\n")
                            continue
                        line_len=len(line.encode("utf-8"))
                        print(line)
                        print(line.encode("utf-8"),line_len)
                        #print(mapping_len, line_len)
                        s=""
                        while index<mapping_len and line_len>0:
                            start, end=token_mapping[index]
                            if sign>0:
                                story_s_file.write(" ")
                            story_s_file.write(text[start:end+1].decode("utf8"))
                            s+=text[start:end+1].decode("utf8")
                            
                            sign=1
                            line_len-=end-start+1
                            print(text[start:end+1].decode("utf8"),end-start+1,end=" ")
                            index+=1
                        story_s_file.write("\n")
                        print(s+"\n")
                except:
                    error_file.write(story_file_name)
                    error_file.flush()
                    print("error:",story_file_name)
                        
                story_s_file.close()
                    
                count+=1
                print("tokenize: %d-%s" % (count, story_file_name))

                story_file.close()
            error_file.close()
            file.close()

    def split():
        count=0
        for file_name in file_names:
            file=open(file_name)
            for line in file.readlines():
                line=line.strip()
                story_t_file=open(common.path_stories_t+"/"+line+".story")
                story_ts_file=open(common.path_stories_ts+"/"+line+".story","w")
                for line in story_t_file.readlines():
                    line=line.strip()
                    #print(line)
                    if line=="":
                        story_ts_file.write("\n")
                    else:
                        story_ts_file.write(line)
                        if line[-1]!=" ":
                            story_ts_file.write(" ")
                story_ts_file.close()
                story_t_file.close()
                count+=1
                print("split: %d" % (count))
            file.close()

    def mention_uniform():
        count=0
        for file_name in file_names:
            file=open(file_name)
            for line in file.readlines():
                line=line.strip()
                mention_mapping=LoadEntityMapping(common.path_mentions+"/"+line+".txt") 
                story_ts_file=open(common.path_stories_ts+"/"+line+".story")
                #story_tsu_file=open(common.path_stories_tsu+"/"+line+".story","w") 
                mention_ex_file=open(common.path_mentions_ex+"/"+line+".txt","w") 
                
                line_starts=[]
                token_index=0
                tokens=[]
                for line in story_ts_file.readlines():
                    line=line.strip()
                    line_tokens=line.split()
                    tokens.extend(line_tokens)
                    line_starts.append(token_index)
                    token_index+=len(line_tokens)
                
                pre_entity_id=-1
                mention_names={}
                for (entity_id, start, end) in mention_mapping:
                    length=0
                    for token in tokens[start:end+1]:
                        length+=len(token)
                    if not  entity_id in mention_names or len(mention_names[entity_id])<length:
                        mention_names[entity_id]=" ".join(tokens[start:end+1])
                
                mentions=[]
                for (entity_id, start, end) in mention_mapping:
                    for line_index in range(len(line_starts)):
                        line_start=line_starts[line_index]
                        if line_start>=start: break
                    #mentions.append((entity_id, mention_names[entity_id], line_index, start, end))
                    mention_ex_file.write("%d\t%s\t%d\t%d\t%d\n" % (entity_id, mention_names[entity_id], line_index, start, end))
                
                mention_ex_file.close()
                #story_tsu_file.close()
                story_ts_file.close()
                count+=1
                print("mention_uniform: %d" % (count))
            file.close()
            
    def entities2mentions():
        count=0
        for file_name in file_names:
            file=open(file_name)
            for line in file.readlines():
                line=line.strip()
                mention_mapping=LoadEntityMapping(common.path_mentions+"/"+line+".txt") 
                story_ts_file=open(common.path_stories_ts+"/"+line+".story")
                entities2mentions_file=open(common.path_entities2mentions+"/"+line,"w") 
                
                line_starts=[]
                token_index=0
                tokens=[]
                for line in story_ts_file.readlines():
                    line=line.strip()
                    line_tokens=line.split()
                    tokens.extend(line_tokens)
                    line_starts.append(token_index)
                    token_index+=len(line_tokens)
                
                #mention_mapping=sorted(mention_mapping, cmp=lambda x,y:x[0]-y[0] or x[1]-y[1])
                entity_mentions={}
                entities=[]
                mention_mapping.sort(key=lambda x:(x[1]))
                for (entity_id, start, end) in mention_mapping:
                    if entity_id not in entities:
                        entities.append(entity_id)
                        entity_mentions[entity_id]=[]
                    
                    entity_mentions[entity_id].append(common.preprocess_line4(" ".join(tokens[start:end+1])))
                    
                for entity in entities:
                    entities2mentions_file.write("%d" % entity)
                    mentions=entity_mentions[entity]
                    for mention in mentions:
                        entities2mentions_file.write("|||")
                        entities2mentions_file.write(mention)
                    entities2mentions_file.write("\n")
                
                entities2mentions_file.close()
                story_ts_file.close()
                count+=1
                print("entities2mentions: %d" % (count))
            file.close()
            
    def separate_text_and_summary():
        count=0
        for file_name in file_names:
            file=open(file_name)
            
            for line in file.readlines():
                is_highlight=False
                line=line.strip()
                story_ts_file=open(common.path_stories_ts+"/"+line+".story")
                texts_ts_file=open(common.path_texts_ts+"/"+line,"w")
                texts_tsl_file=open(common.path_texts_tsl+"/"+line,"w")
                texts_tslr_file=open(common.path_texts_tslr+"/"+line,"w")
                summaries_ts_file=open(common.path_summaries_ts+"/"+line,"w")
                summaries_tsl_file=open(common.path_summaries_tsl+"/"+line,"w")
                summaries_tslr_file=open(common.path_summaries_tslr+"/"+line,"w")
                for line in story_ts_file.readlines():
                    line=line.replace('\n', '').replace('\r', '')
                    if line=='@ highlight ':
                        is_highlight=True
                        continue
                    
                    if not is_highlight:
                        texts_ts_file.write(line)
                        line=common.preprocess_line3(line)
                        texts_tsl_file.write(line+"\n")
                        texts_tslr_file.write(line)
                    else:
                        summaries_ts_file.write(line)
                        line=common.preprocess_line3(line)
                        summaries_tsl_file.write(line+"\n")
                        line=line.strip()
                        if line[-1]!='.' or line[-1]!='?' or line[-1]!='!' or line[-1]!=':': 
                            line=line+" . "
                        summaries_tslr_file.write(line)
                story_ts_file.close()
                texts_ts_file.close()
                texts_tsl_file.close()
                texts_tslr_file.close()
                summaries_ts_file.close()
                summaries_tsl_file.close()
                summaries_tslr_file.close()
                count+=1
                print("separate: %d" % (count))
            file.close()
            
    def add_dot_to_summary():
        count=0
        for file_name in file_names:
            file=open(file_name)
            
            for line in file.readlines():
                line=line.strip()
                summaries_tsl_file=open(common.path_summaries_tsl+"/"+line,"r")
                summaries_tslr_file=open(common.path_summaries_tslr+"/"+line,"w")
                for line in summaries_tsl_file.readlines():
                    line=line.strip()
                    if line=="":continue
                    if line[-1]!='.' and line[-1]!='?' and line[-1]!='!' and line[-1]!=':': 
                        line=line+" . "
                    summaries_tslr_file.write(line+"\n")
                summaries_tsl_file.close()
                summaries_tslr_file.close()
                count+=1
                print("add_dot_to_summary: %d" % (count))
            file.close()

    #tokenize()
    #split()
    #mention_uniform()
    #entities2mentions()
    #separate_text_and_summary()
    add_dot_to_summary()


def process_entity_embeddings():
    
    file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
    #file_names=[common.path_train_file]
    
    def entity_linking():
        #run codes in Java
        pass
    
    def create_entity_vocab():
        entity_counts={}
        for file_name in file_names:
            file=open(file_name)
            for line in file.readlines():
                line=line.strip()
                    
                file_entities = open(common.path_entities+"/"+line)
                for line in file_entities:
                    line=line.strip()
                    items=line.split("\t")
                    if not items[1] in entity_counts:
                        entity_counts[items[1]]=0
                    entity_counts[items[1]]+=1
                file_entities.close()
                
            file.close()
        
        file_entity_vocab=open(common.path_entity_vocab, "w")
        entities=list(entity_counts.keys())
        entities.sort(key=lambda x:entity_counts[x], reverse=True)
        entities=entities[:500000]
        for entity in entities:
            file_entity_vocab.write("%s\t%d\n" % (entity, entity_counts[entity]))
        file_entity_vocab.close()
    
    def create_entity_embedding():
        kg = rdflib_to_kg(common.path_kg,filetype="n3")
        random_walker = RandomWalker(2, float('inf'))
        transformer = RDF2VecTransformer(vector_size=128, walkers=[random_walker], sg=1)
        
        train_entities=[]
        file_entities=open(common.path_entity_vocab)
        for line in file_entities.readlines():
            line=line.strip()
            train_entities.append("<"+line.split("\t")[0]+">")
        file_entities.close()
        
        entity_embeddings = np.array(transformer.fit_transform(kg, train_entities))
        print(len(entity_embeddings), len(train_entities))
        if len(entity_embeddings)==len(train_entities):
            entity_embeddings.tofile(common.path_entity_embeddings)

    def test():
        a=np.fromfile(common.path_entity_embeddings, np.float32)
        a=np.reshape(a,[-1,128])
        print(len(a))
    
    #282827
    #entity_linking()
    create_entity_vocab()
    create_entity_embedding()
    #test()


def process_word_embedding():
    
    def tokenize_wiki_text():
        from nltk.tokenize.stanford import StanfordTokenizer 
        tokenizer = StanfordTokenizer(path_to_jar=common.path_parser)
        file = open("/home/test/kbsumm/data/wiki2text-master/enwiki.txt")
        w_file = open("/home/test/kbsumm/data/wiki2text-master/enwiki-tokenized.txt", "w")
        count=0
        while True:
            line=file.readline()
            if not line:
                break
            line=line.strip()
            if line !="":
                w_file.write(common.preprocess_line(tokenizer, line)+"\n")
                count+=1
                print(count)
        w_file.close()
        file.close()
        
    def tokenize_wiki_text2():
        import nltk
        file = open("/home/test/kbsumm/data/wiki2text-master/enwiki.txt")
        w_file = open("/home/test/kbsumm/data/wiki2text-master/enwiki-tokenized.txt", "w")
        count=0
        while True:
            line=file.readline()
            if not line:
                break
            line=line.strip()
            if line !="":
                line=" ".join(nltk.word_tokenize(line))
                #print(line)
                w_file.write(common.preprocess_line2(line)+"\n")
                #print(common.preprocess_line2(line))
                count+=1
                print(count)
        w_file.close()
        file.close()
        
    def merge_text():
        count=0
        w_file = open("/home/test/kbsumm/data/wiki2text-master/wiki-dm-cnn.txt", "a")
        #file = open("/home/test/kbsumm/data/wiki2text-master/enwiki-tokenized.txt")
        #line=file.readline()
        #w_file.write(line.lower())
        #while True:
        #    line=file.readline()
        #    if not line:
        #        break
        #    line=line.strip().lower()
        #    w_file.write(" ")
        #    w_file.write(line)
        #    count+=1
        #    print(count)
        #file.close()
        
        file_names=[common.path_base+"/data/dailymail/texts-tslr", 
                    common.path_base+"/data/dailymail/summaries-tslr",
                    common.path_base+"/data/cnn/texts-tslr",
                    common.path_base+"/data/cnn/summaries-tslr"]
        for file_name in file_names:
            fns=os.listdir(file_name)
            for fn in fns:
                file = open(file_name+"/"+fn)
                text=file.read().strip()
                w_file.write(" ")
                w_file.write(text)
                file.close()
                count+=1
                print("dm-cnn:",count)
        
        w_file.close()
        
    #file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
    #file_names=[common.path_train_file]
    
    def build_word_vocab():
        data=[]
        file_names=os.listdir(common.path_texts_tsl)
        for file_name in file_names:
            with open(common.path_texts_tsl+'/'+file_name, 'r') as file:
                words=file.read().replace('\n','').split()
                data.extend(words)
                
        file_names=os.listdir(common.path_summaries_tsl)
        for file_name in file_names:
            with open(common.path_summaries_tsl+'/'+file_name, 'r') as file:
                words=file.read().replace('\n','').split()
                data.extend(words)
    
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        word_list=[word for word, count in count_pairs]
        
        count=0 
        with open(common.path_word_vocab, 'w') as vocab:
            for item1, item2 in count_pairs:
                if count==50000:
                    break
                print(item1, item2)
                vocab.write('%s\t%s\n' % (item1, item2))
                count=count+1
                
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
    
    #tokenize_wiki_text2()
    #build_word_vocab()
    #merge_text()
    build_word_vocab()
    build_total_embeddings()
    vocab_embedding()


def build_ext_gold_standard():
    import rouge
    import struct
    
    rouge=rouge.Rouge()
    error_file=open(common.path_corpus+"/train-error", "w")
    def rouge_score(scores):
        return scores['rouge-2']['f']
    def greedy(lines, ref_summary):
        summary=''
        summary_lines=[]
        summary_indice=[]
        summary_score=0
        finished=False
        while not finished:
            finished=True
            max_increment=0
            max_line=''
            max_summary_score=0
            max_i=0
            
            for i in range(len(lines)):
                line=lines[i]
                tmp_summary=summary+line
                tmp_summary_score=rouge_score(rouge.get_scores(tmp_summary,ref_summary)[0])
                if max_increment<tmp_summary_score-summary_score:
                    max_increment=tmp_summary_score-summary_score
                    max_line=line
                    max_summary_score=tmp_summary_score
                    max_i=i
                    finished=False
                    
            if not finished:
                summary+=max_line
                summary_lines.append(max_line)
                summary_indice.append(max_i)
                summary_score=max_summary_score
                lines[max_i]=''

        return summary, summary_lines, summary_score, summary_indice
    
    def Mapper(t):
        file_name,n=t
        with open(common.path_summaries_tsl+'/'+file_name) as file:
            abs_summary=file.read().replace('\n','. ')
        with open(common.path_texts_tsl+'/'+file_name) as file:
            text=file.read().strip('\n')
            lines=text.split('\n')
            lines2=[]
            for line in lines:
                line=line.strip("\n").strip(" ")
                if line=="": continue
                lines2.append(line)
        try:
            summary, summary_lines, summary_score, summary_indice=greedy(lines2,abs_summary)
        except:
            error_file.write(file_name+"\n")
            error_file.flush()
            return False
        
        labels=[]
        with open(common.path_summaries_ext_tsl+'/'+file_name,'w') as wfile:
            wfile.write('\n'.join(summary_lines))
        with open(common.path_summaries_label_tsl+'/'+file_name,'wb') as wfile:
            for _ in range(len(lines2)):
                labels.append(0)
            for item in summary_indice:
                labels[item]=1
            for label in labels:
                wfile.write(struct.pack('f',label))
                
        #print(text)
        print(n, file_name,'-----------------------------%s----------------------------' % file_name, summary_indice)
                                
        #print()
        return True
    
    def Build(request_parallelism=5):
        #abs_summaries=os.listdir(common.path_summaries_tsl)
        #with open(common.path_test_test_file) as file:
        #    abs_summaries=[line.strip() for line in file.readlines()]
        #exist_summaries=os.listdir(common.path_summaries_ext_tsl)
        #abs_summaries=[file_name for file_name in abs_summaries if file_name not in exist_summaries]
        abs_summaries=[]
        file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
        #file_names=[common.path_corpus+"/trains-error"]
        count=0
        for file_name in file_names:
            
            with open(file_name) as file:
                for line in file.readlines():
                    #if count>1000:break
                    abs_summaries.append(line.strip())
                    count+=1
                    
        #abs_summaries=abs_summaries[:1000]
        print(len(abs_summaries))
        p = ThreadPool(request_parallelism)
        results = p.imap_unordered(Mapper, zip(abs_summaries, range(len(abs_summaries))))
        i=0
        try:
            for ok in results:
                i=i+1
        except KeyboardInterrupt:
            print ('Interrupted by user')
                    
    Build()
    error_file.close()
    
    
def build_ext_gold_standard_2():
    import rouge
    import struct
    
    rouge=rouge.Rouge()
    error_file=open(common.path_corpus+"/train-error", "w")
    def rouge_score(scores):
        return scores['rouge-l']['f']
    def greedy(text_lines, ext_indices, ref_summary, file_name):
        summary_lines=[]
        
        for i in range(len(text_lines)):
            line=text_lines[i].strip("\n").strip(" ")
            if line=='':
                continue
            try:
                if ext_indices[i]==0:
                    scores=rouge.get_scores([line] * len(ref_summary),ref_summary)
                    scores.sort(key=lambda x:x['rouge-l']['f'], reverse=True)
                    score=scores[0]['rouge-l']['f']
                    if score>=0.5:
                        summary_lines.append(line)
                        ext_indices[i]=1
            except:
                error_file.write("greedy:"+file_name+"\n")
                error_file.flush()
    
    def Mapper(t):
        file_name,n=t
        with open(common.path_summaries_tsl+'/'+file_name) as file:
            ref_summary=[]
            for line in file.readlines():
                line=line.strip("\n").strip(" ")
                if line!="":
                    ref_summary.append(line)
        with open(common.path_texts_tsl+'/'+file_name) as file:
            text=file.read().strip('\n')
            lines=text.split('\n')
            text_lines=[]
            for line in lines:
                line=line.strip("\n").strip(" ")
                if line=="": continue
                text_lines.append(line)
        try:
            ext_indices=np.fromfile(common.path_summaries_label_tsl+'/'+file_name, np.float32)
        except:
            error_file.write(file_name+"\n")
            return
        len1=np.sum(ext_indices)
        #print(file_name, len(text_lines), len(ext_indices))
        #print(ext_indices)
            
        greedy(text_lines,ext_indices,ref_summary,file_name)
        with open(common.path_summaries_ext_tsl_2+'/'+file_name,'w') as wfile:
            for i in range(len(ext_indices)):
                if ext_indices[i]==1:
                    wfile.write(text_lines[i]+"\n")
        #with open(common.path_summaries_label_tsl_2+'/'+file_name,'wb') as wfile:
        ext_indices.tofile(common.path_summaries_label_tsl_2+'/'+file_name)
        len2=np.sum(ext_indices)
                
        #print(text)
        #print(n, file_name,'----%s----' % file_name, len1, len2)
        if len1!=len2:print(file_name, len1, len2)
                                
        #print()
        return True
    
    def Build(request_parallelism=5):
        #abs_summaries=os.listdir(common.path_summaries_tsl)
        abs_summaries=[]
        file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
        count=0
        for file_name in file_names:
            with open(file_name) as file:
                if count>1000:break
                abs_summaries.extend([line.strip() for line in file.readlines()])
                count+=1
        #exist_summaries2=os.listdir(common.path_summaries_ext_tsl_2)
        #abs_summaries=[file_name for file_name in abs_summaries if file_name not in exist_summaries2]
        
        print(len(abs_summaries))
        p = ThreadPool(request_parallelism)
        results = p.imap_unordered(Mapper, zip(abs_summaries, range(len(abs_summaries))))
        i=0
        try:
            for ok in results:
                i=i+1
        except KeyboardInterrupt:
            print ('Interrupted by user')
                    
    Build()
    error_file.close()
    
    def process_if_no_ext_1():
        w_file=open(common.path_corpus+"/process_if_no_ext", "w")
        file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
        for file_name in file_names:
            with open(file_name) as file:
                for one_data_file_name in file.readlines():
                    one_data_file_name=one_data_file_name.strip()
                    one_data_file=open(common.path_summaries_ext_tsl_2+"/"+one_data_file_name)
                    
                    text=one_data_file.read()
                    if text.strip("\n").strip(" ")=="":
                        w_file.write(one_data_file_name+"\n")
                        
                    one_data_file.close()
        w_file.close()       
        
    def process_if_no_ext_2():
        with open(common.path_corpus+"/process_if_no_ext") as file:
            for file_name in file.readlines():
                file_name=file_name.strip()
                one_data_file=open(common.path_summaries_ext_tsl_2+"/"+file_name, "w")
                one_source_file=open(common.path_texts_tsl+"/"+file_name)
                
                source_lines=one_source_file.readlines()
                source_lines=[source_line.strip("\n").strip(" ") for source_line in source_lines]
                
                
                published_index=-1
                updated_index=-1
                for i in range(len(source_lines)):
                    if source_lines[i]!="":
                        if source_lines[i]=="published :":published_index=i
                        if source_lines[i]=="updated :":updated_index=i
                
                count=0
                if updated_index>-1:index=updated_index+2
                elif published_index>-1:index=published_index+2
                else:index=0
                for i in range(index, len(source_lines)):
                    if count==3:break
                    if source_lines[i] != "":
                        one_data_file.write(source_lines[i]+"\n")
                        count+=1
                
                one_source_file.close()
                one_data_file.close()
        
    #process_if_no_ext_1()    
    #process_if_no_ext_2()   


def entity_gold_standard():
    data_files=[common.path_train_file, common.path_dev_file, common.path_test_file]
    
    def phase1():
        count=0
        for data_file in data_files:
            data_file = open(data_file)
            
            for line in data_file.readlines():
                file_line=line.strip()
                
                story_ts_file=open(common.path_stories_ts+"/"+file_line+".story")
                summary_index=0
                for line in story_ts_file.readlines():
                    if line.strip(" ").strip("\n").strip("\r")=="@ highlight ":
                        #print(summary_index)
                        break
                    summary_index+=1
                story_ts_file.close()
                
                key_entities=[]
                entities_positions={}
                mentions_ex=open(common.path_mentions_ex+"/"+file_line+".txt")
                for line in mentions_ex.readlines():
                    line=line.strip()
                    items=line.split("\t")
                    if items[0] not in entities_positions:
                        entities_positions[items[0]]=[]
                    position=int(items[2])
                    #print(summary_index, position, items[0])
                    if position>summary_index and items[0] not in key_entities:
                        key_entities.append(items[0])
                    entities_positions[items[0]].append(position)
                mentions_ex.close()
                #print(key_entities)
                
                true_label=false_label=all_lines=0
                entities2mentions=open(common.path_entities2mentions+"/"+file_line)
                entities2mentions_labeled=open(common.path_entities2mentions_labeled+"/"+file_line, "w")
                for line in entities2mentions.readlines():
                    line=line.strip()
                    items=line.split("|||")
                    all_lines+=1
                    #print(items)
                    positions=entities_positions[items[0]]
                    if items[0] in key_entities:
                        for position in positions:
                            if position<=summary_index:
                                entities2mentions_labeled.write(line+"|||1\n")
                                true_label+=1
                                break
                    else:
                        for position in positions:
                            if position<=summary_index:
                                entities2mentions_labeled.write(line+"|||0\n")
                                false_label+=1
                                break
                entities2mentions_labeled.close()
                entities2mentions.close()
                
                count+=1
                print(count, summary_index, true_label, false_label, all_lines-true_label-false_label, file_line)
            
            data_file.close()
    
    def phase2():
        count=0
        true_label=0
        for data_file in data_files:
            data_file = open(data_file)
            
            for line in data_file.readlines():
                count+=1
                file_line=line.strip()
                with open(common.path_entities2mentions_labeled+"/"+file_line) as file:
                    lines=file.readlines()
                
                for line in lines:
                    line=line.strip()
                    items=line.split("|||")
                    if items[-1]=="1": true_label+=1
                    
                if true_label == 0: print(file_line)
        
        print(true_label/count)
    
    def phase3():
        file_lines=["2fbee5b28cb71aa6c72e87ebee812cd2597adc5d", "4adfdfb919a8afefdf981741090c03ae7e8f260d", "3ffe69025058a3ac8a1c237334f67879501cf571",
                    "9b668819e5bf2a3b7ee6fa1bddc8890c7a2ce2ee", "9877a59059dde5d2767a826ff5f15816097550e5", "6714f185ff0cb35a46b34d782ac14d893a048c37", 
                    "ace583eb50523061c4e9b7de2893566f2982aba0", "445a4150e2d465c5569efa68b2fec9a4781b5d91", "bbf51611ff1b569b97f27e8394e0bc1e7c65e2fc",
                    "ec1238044c9c3f01b22585d27eacfdf92ddc83d0", "2806217f9e8ee1e9fa91819f1c048956eedec012", "369c7b9f0dea275ce23545ee178b74f0d57f87e3",
                    "e75cf878246327a14c35021b929eba37cc7c2e3b", "e8e77f9626117582a2ed37aa1b336df608bd0f71", "eb40c55253c0e9aa97529f363a1678e12bd0b3ad",
                    "8756bc41bc29767498831d9044e30cda5a8f50b9", "c5709a2f8699728b8950c3ff74d3ab8d416be235", "ace13f441c2187bd204f18d553551c83805b6b16",
                    "50b524546cf695dda95b2578b59fceb09482b5f7", "fd259d0d7d18a280b373b4e857a476889d47d218", "2d60847f2df71016539d2abf7564e0d70dd33b58",
                    "e735ccd620cd037e33494c6507ae395dc6215073", "1a0986dcb2e051197ae9229fdb731a254d1bc7fa", "14e107a4dab734820ebef24c563488b48222049a",
                    "c02974e2da600558fc202d87e28dfa003184dc7d", "1345ef9993782bf71b923ef4eff363dd237b9a61", "8defb1a7e2a480534852b213248f5c061cf1fd84",
                    "64208179ea8912eac9605a1237655277ebe47b66", "34aabe1eb17510b98a7528cc4fc2ea5f3299ec21", "3582ad9c86deeea61ed853379a1b52e372dd8c9d",
                    "3d0b9af92205ad6335c35986360d62e0a50c0bf4", "79166b1f5f8f6e09a4807223f093f6d929213f30", "49345a2a4ee45c3be06a8c002f20f8eda03d515a",
                    "5df07131d304dd9da610dd83c48a84f3fb47de74", "3ef06c068074e1ffc4b6215f1c96d43f428e459f"]
        
        avg_entity=4
        count=0
            
        for line in file_lines:
            file_line=line.strip()
                
            true_label=0
            with open(common.path_entities2mentions_labeled+"/"+file_line) as file:
                lines=[]
                for line in file.readlines():
                    line=line.strip()
                    if line!="":
                        lines.append(line)
            
            mentions_count=[]
            for i in range(len(lines)):
                items=lines[i].strip().split("|||")
                if items[-1]=="1": true_label+=1
                mentions_count.append((i, len(items[1:-1])))
                
            mentions_count.sort(key=lambda x:-x[1])
            for i in range(avg_entity):
                line=lines[mentions_count[i][0]]
                lines[mentions_count[i][0]]=line[:line.rindex("|||")]+"|||1"
            with open(common.path_entities2mentions_labeled+"/"+file_line, "w") as w_file:
                for line in lines:
                    w_file.write(line+"\n")
                
            count+=1
            print(count, file_line)

    #phase1()
    phase2()
    #phase3()
    
def clear_errors():
    data_files=[common.path_train_file, common.path_dev_file, common.path_test_file]
    
    def clear_empty_text():
        error_file=open(common.path_corpus+"/train-error", "w")
        for data_file in data_files:
            data_file = open(data_file)
            
            for line in data_file.readlines():
                file_line=line.strip()
                with open(common.path_texts_tsl+"/"+file_line) as text_file:
                    text=text_file.read().strip(" ").strip("\n")
                if text=="":error_file.write("%s\n" % file_line)
                
        error_file.close()

    clear_empty_text()

def merge_dm_cnn():
    from shutil import copyfile
    
    path_dm_cnn = common.path_base + "/data/cnn+dm"
    path_dm = common.path_base + "/data/dailymail"
    path_cnn = common.path_base + "/data/cnn"
    
    path_tokens="/tokens"
    path_stories="/stories"
    path_stories_t="/stories-t"
    path_stories_ts="/stories-ts"
    path_stories_tsa="/stories-tsa"
    
    path_mentions="/mentions"
    path_mentions_ex="/mentions-ex"
    
    path_entities="/entities"
    path_entities2mentions="/entities2mentions"
    path_entities2mentions_labeled="/entities2mentions-labeled"
    
    path_texts_ts="/texts-ts"
    path_texts_tsl="/texts-tsl"
    path_texts_tslr="/texts-tslr"
    path_texts_tsla="/texts-tsla"
    
    path_summaries_ts="/summaries-ts"
    path_summaries_tsl="/summaries-tsl"
    path_summaries_tslr="/summaries-tslr"
    path_summaries_tsla="/summaries-tsla"
    
    path_summaries_ext_tsl="/summaries-ext-tsl"
    path_summaries_label_tsl="/summaries-label-tsl"
    
    path_summaries_ext_tsl_2="/summaries-ext-tsl-2"
    path_summaries_label_tsl_2="/summaries-label-tsl-2"
    
    path_corpuses = [path_cnn]
    #path_data_files = ["trains", "devs", "tests"]
    path_dirs = [path_tokens, path_stories, path_stories_t, path_stories_ts, path_stories_tsa, 
                 path_mentions, path_mentions_ex, path_entities, path_entities2mentions, path_entities2mentions_labeled,
                 path_texts_ts, path_texts_tsl, path_texts_tslr, path_texts_tsla,
                 path_summaries_ts, path_summaries_tsl, path_summaries_tslr, path_summaries_tsla,
                 path_summaries_ext_tsl, path_summaries_label_tsl, path_summaries_ext_tsl_2, path_summaries_label_tsl_2]
    
    for path_dir in path_dirs:
        path_dir = path_dm_cnn + path_dir
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
    
    for path_corpus in path_corpuses:
        for path_dir in path_dirs:
            print(path_corpus, path_dir)
            path_dir2 = path_corpus + path_dir
            path_dir3 = path_dm_cnn +path_dir
            file_names = os.listdir(path_dir2)
            for file_name in file_names:
                copyfile(path_dir2 + "/" +file_name, path_dir3 + "/" +file_name)
                    

def statistics():
    data_files=[common.path_train_file, common.path_dev_file, common.path_test_file]
    def D_L():
        dl=dn=0
        for path_data in data_files:
            with open(path_data) as file:
                for fn in file.readlines():
                    fn=fn.strip()
                    tfile=open(common.path_texts_tsl+"/"+fn)
                    text=tfile.read()
                    text=text.strip().replace("\n"," ")
                    words=text.split(" ")
                    tfile.close()
                    dl+=len(words)
                    dn+=1
                    #print(fn)
        print("D.L=%f" % (dl/dn))
        
    def S_L():
        l=n=0
        for path_data in data_files:
            with open(path_data) as file:
                for fn in file.readlines():
                    fn=fn.strip()
                    tfile=open(common.path_summaries_tsl+"/"+fn)
                    text=tfile.read()
                    text=text.strip().replace("\n"," ")
                    words=text.split(" ")
                    tfile.close()
                    l+=len(words)
                    n+=1
        print("S.L=%f" % (l/n))
        
    def D_E_M():
        dn=en=mn=mn2=sen=eyago=0
        for path_data in data_files:
            with open(path_data) as file:
                for fn in file.readlines():
                    fn=fn.strip()
                    all_entities={}
                    key2keys={}
                    with open(common.path_entities + "/" + fn) as file: 
                        for line in file.readlines():
                            line=line.strip()
                            items=line.split("\t")
                            all_entities[items[0]]=items[2]
                            if items[-1]!="-1":
                                eyago+=1
                            keys=items[0].strip("+").split("+")
                            for key in keys:
                                key2keys[key]=items[0]
                  
                    mentions = []
                    mentions2 = []
                    entities2mentions = {}
                    entity_ids=[]
                    with open(common.path_entities2mentions_labeled + "/" + fn) as file: 
                        for line in file.readlines():
                            line=line.strip()
                            items=line.split("|||")
                            entity_ids.append(items[0])
                            entities2mentions[items[0]]=items
                  
                    entities = []
                    entity_labels=[]
                    entity_orders=[]
                    try:
                        for entity_id in entity_ids:
                            if entity_id not in entity_orders:
                                entity_labels.append(int(entities2mentions[entity_id][-1]))
                                keys=key2keys[entity_id]
                                entities.append(all_entities[keys])
                                keys=keys.strip("+").split("+")
                                ss=[]
                            for key in keys:
                                if key in entities2mentions:
                                    entity_orders.append(key)
                                    if entities2mentions[key][-1]=="1":
                                        entity_labels[-1]=1
                                    ss.extend(entities2mentions[key][1:-1])
                            mentions.append(np.unique(np.array(ss)))
                            mentions2.append(np.array(ss))
                    except:
                        print("e")
                        continue
                    entity_labels=np.array(entity_labels, dtype=np.int32)    
                    en+=len(entity_labels)
                    sen+=np.sum(entity_labels)
                    mn+=sum([len(m) for m in mentions])
                    mn2+=sum([len(m) for m in mentions2])
                    dn+=1
                    
        print("E.N=%f" % (en/dn))
        print("E.L=%f" % (mn/en))
        print("E.L2=%f" % (mn2/en))
        print("S.E.N=%f" % (sen/dn))
        print("E.YAGO=%f" % (eyago/dn))
        
    def E_N_2():
        l=n=0
        for path_data in data_files:
            with open(path_data) as file:
                for fn in file.readlines():
                    fn=fn.strip()
                    tfile=open(common.path_entities+"/"+fn)
                    lines=tfile.readlines()
                    tfile.close()
                    l+=len(lines)
                    n+=1
        print("E.N=%f" % (l/n))
        
    def D_Sents():
        l=n=0
        for path_data in data_files:
            with open(path_data) as file:
                for fn in file.readlines():
                    fn=fn.strip()
                    tfile=open(common.path_texts_tsl+"/"+fn)
                    lines=tfile.readlines()
                    tfile.close()
                    l+=len(lines)
                    n+=1
        print("D.Sents=%f" % (l/n))
        
    def S_Sents():
        l=n=0
        for path_data in data_files:
            with open(path_data) as file:
                for fn in file.readlines():
                    fn=fn.strip()
                    tfile=open(common.path_summaries_tsl+"/"+fn)
                    lines=tfile.readlines()
                    tfile.close()
                    l+=len(lines)
                    n+=1
        print("S.Sents=%f" % (l/n))
    
    #D_L()
    #S_L()
    #D_E_M()
    #E_N_2()
    D_Sents()
    S_Sents()

def test():
    kg = rdflib_to_kg("/home/test/kbsumm/data/test.tsv",filetype="n3")
    random_walker = RandomWalker(2, float('inf'))
    transformer = RDF2VecTransformer(vector_size=256, walkers=[random_walker], sg=1)
    train_people = ["<yagoTheme_yagoFacts>", "<Fernando_Romeo_Lucas_García>"]
    walk_embeddings = transformer.fit_transform(kg, train_people)
    walk_embeddings = np.array(walk_embeddings)
    walk_embeddings.tofile(common.path_entity_embeddings)
    print(walk_embeddings)
    

def test2():
    file=open("/home/test/kbsumm/aidalight/data/bned/resources/id_entity")
    for line in file:
        line=line.strip(" ").strip("\n").strip("\r")
        item=line.split("\t")[1]
        #print(item)
        if item=="British_Columbia":
            print(line)
    file.close()
    
def test3():
    m={"a":4,"b":3,"c":5}
    a=list(m.keys())
    a.sort(key=lambda x:m[x])
    print(a)


def test4():
    count=0
    with open("/home/test/kbsumm/data/wiki2text-master/README.md") as file:
        while True:
            line=file.readline()
            if not line:
                break
            print(line.strip())
            
def test5():
    abs_summaries=os.listdir(common.path_summaries_tsl)
    for abs_summary in abs_summaries:
        if not os.path.exists(common.path_summaries_ext_tsl+"/"+abs_summary):
            print(abs_summary)
            
def test6():
    import re
    a="a    b  c   d           e"
    a=re.sub("\s+", " ", a)
    print(a)


def test7():
    a="0|||mark duell"
    print(a[:a.rindex("|||")])


def test8():
    file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
    error_file=open(common.path_corpus+"/trains-error", "w")
    for f in file_names:
        with open(f) as file:
            for file_name in file.readlines():
                file_name=file_name.strip()
                try:
                    with open(common.path_texts_tsl+'/'+file_name) as file:
                        text=file.read().strip('\n')
                        lines=text.split('\n')
                        text_lines=[]
                        for line in lines:
                            line=line.strip("\n").strip(" ")
                            if line=="": continue
                            text_lines.append(line)
                        ext_indices=np.fromfile(common.path_summaries_label_tsl+'/'+file_name, np.float32)
                    if len(ext_indices)!=len(text_lines):error_file.write(file_name+"\n")
                except:
                    error_file.write(file_name+"\n")
    error_file.close()



if __name__ == '__main__':
    #create_train_dev_test()
    #story_preprocess()
    #import tkinter
    #process_entity_embeddings()
    #process_word_embedding()
    #exit()
    #build_ext_gold_standard()
    #build_ext_gold_standard_2()
    #entity_gold_standard()
    #clear_errors()
    #merge_dm_cnn()
    statistics()
    #test()21 f6d4d979a5645f10011b74fab10f6fac7833b09d -----------------------------f6d4d979a5645f10011b74fab10f6fac7833b09d-------------
    #test2()
    #test3()
    #test4()
    #test5()
    #test6()
    #test7()
    #test8()
















