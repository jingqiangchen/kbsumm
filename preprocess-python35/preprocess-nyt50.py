# -*- coding: utf-8
import os
import common
from NYTDoc import NYTDoc 
from readline import read_history_file
import rdflib
import pandas as pd
from converters import rdflib_to_kg
from rdf2vec import RDF2VecTransformer
from multiprocessing.pool import ThreadPool, Pool
#from itertools import izip
from itertools import repeat

import numpy as np
import collections

from walkers import (RandomWalker, WeisfeilerLehmanWalker, 
                     AnonymousWalker, WalkletWalker, NGramWalker,
                     CommunityWalker, HalkWalker)
from pandas._libs.tslibs.offsets import key
from numpy import source

def berkeley_sents():
    def collect_sents(indir, outdir):
        filenames=os.listdir(indir)
        filenames.sort()
        for filename in filenames:
            infile=open(indir+"/"+filename)
            text_outfile=open(outdir+"/"+filename,"w")
            lines=infile.readlines()[1:-1]
            words=[]
            for line in lines:
                line=line.strip()
                if line=="":
                    text_outfile.write(" ".join(words))
                    text_outfile.write("\n")
                    words.clear()
                    continue
                items=line.split("\t")
                words.append(items[3])
            text_outfile.close()
            infile.close()
    
    collect_sents(common.path_bekerlay_train1, common.path_bekerlay_texts)
    collect_sents(common.path_bekerlay_train2, common.path_bekerlay_summaries)
    collect_sents(common.path_bekerlay_eval1, common.path_bekerlay_texts)
    collect_sents(common.path_bekerlay_eval2, common.path_bekerlay_summaries)

def berkeley_mentions():
    import re
    #GPE, PERSON, ORG, EVENT
    def collect_mentions(indir, outdir):
        filenames=os.listdir(indir)
        filenames.sort()
        for filename in filenames:
            print(filename)
            infile=open(indir+"/"+filename)
            
            lines=infile.readlines()[1:-1]
            entities={}
            line_num=len(lines)
            i=0
            word_count=word_count2=0
            sent_count=0
            max_entity_id=0
            while i<line_num:
                line=lines[i].strip()
                if line=="":
                    i+=1
                    word_count+=word_count2
                    sent_count+=1
                    continue
                
                items=line.split("\t")
                word_count2=int(items[2])+1
                if items[-2].find("(GPE")>-1 or items[-2].find("(PERSON")>-1 or items[-2].find("(ORG")>-1 or items[-2].find("(EVENT")>-1:
                    mention_position=i
                    j=i
                    #print(filename, items[-2], j, "a:"+lines[j].strip().split("\t")[-1])
                    try:
                        while lines[j].strip().split("\t")[-1][0]!="(": 
                            j-=1
                        mention_start=j
                    except:
                        i+=1
                        continue
                    
                    items2=lines[j].strip().split("\t")
                    #print(items2)
                    m=re.match("^\\((\d+)*", items2[-1])
                    mention=[]
                    entity_id=m.group(1)
                    if entity_id.strip()!="":
                        if max_entity_id<int(entity_id):
                            max_entity_id=int(entity_id)
                    if entity_id not in entities:
                        entities[entity_id]=[]
                    while True:
                        items2=lines[j].strip().split("\t")
                        mention.append(items2[3])
                        j+=1
                        if items2[-1].find(entity_id+")")>-1:
                            mention_end=j-1
                            break
                    entities[entity_id].append([mention,sent_count+1, word_count+mention_start+1, word_count+mention_end+1])
                    if mention_position>=mention_start and mention_position<=mention_end:
                        i=j
                    else:
                        entities[entity_id].pop()
                        i=mention_position
                i+=1
            
            text_outfile=open(outdir+"/"+filename,"w")
            for key in entities:
                entities[key].sort(key=lambda x:(x[1],x[2],x[3]))
            keys=list(entities.keys())
            i=0
            while i<len(keys):
                if len(entities[keys[i]])==0:
                    del keys[i]
                else:i+=1
            keys.sort(key=lambda x:(int(entities[x][0][1]),int(entities[x][0][2])))
            for key in keys:
                mmm=entities[key]
                mmm.sort(key=lambda x:(x[1],x[2],x[3]))
                if len(mmm)>0:
                    for mm in mmm:
                        if len(mm[0])<=5:
                            text_outfile.write("%s\t%s\t%d\t%d\t%d" % (key," ".join(mm[0]), mm[1], mm[2], mm[3]))
                            text_outfile.write("\n")
            text_outfile.close()
            
            infile.close()
            
    collect_mentions(common.path_bekerlay_train1, common.path_bekerlay_text_mentions)
    collect_mentions(common.path_bekerlay_train2, common.path_bekerlay_summary_mentions)
    collect_mentions(common.path_bekerlay_eval1, common.path_bekerlay_text_mentions)
    collect_mentions(common.path_bekerlay_eval2, common.path_bekerlay_summary_mentions)

def create_file_list():
    file_names=os.listdir(common.path_bekerlay_train1)
    file_names.sort()
    with open(common.path_train_file, "w") as wfile:
        for file_name in file_names[:-4000]:
            wfile.write(file_name+"\n")
    
    with open(common.path_dev_file, "w") as wfile:
        for file_name in file_names[-4000:]:
            wfile.write(file_name+"\n")
            
    with open(common.path_test_file) as rfile:
        for fn in rfile.readlines():
            fn=fn.strip()
            with open(common.path_bekerlay_summaries+"/"+fn) as tfile:
                lines=[]
                for line in tfile.readlines():
                    lines.append(line.strip())
                text=" ".join(lines)
                print(len(text.split()))
                
def process_stories():
    fn_file_paths=[common.path_train_file, common.path_dev_file, common.path_test_file]
    def create_stories_texts_summaries():
        starts_file=open(common.path_starts_file, "w")
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    text_file=open(common.path_bekerlay_texts+"/"+fn)
                    summary_file=open(common.path_bekerlay_summaries+"/"+fn)
                    story_file_ts=open(common.path_stories_ts+"/"+fn, "w")
                    text_file_ts=open(common.path_texts_ts+"/"+fn, "w")
                    summary_file_ts=open(common.path_summaries_ts+"/"+fn, "w")
                    
                    lines=text_file.readlines()
                    starts_file.write("%s\t%d\n" % (fn, len(lines)+1))
                    for line in lines:
                        story_file_ts.write(line.strip()+"\n")
                        text_file_ts.write(line.strip()+"\n")
                    for line in summary_file.readlines():
                        story_file_ts.write(line.strip()+"\n")
                        summary_file_ts.write(line.strip()+"\n")
                    
                    summary_file_ts.close()
                    text_file_ts.close()
                    story_file_ts.close()
                    summary_file.close()
                    text_file.close()
        starts_file.close()         
    
    def tslr():
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    text_file_ts=open(common.path_texts_ts+"/"+fn)
                    summary_file_ts=open(common.path_summaries_ts+"/"+fn)
                    text_file_tsl=open(common.path_texts_tsl+"/"+fn, "w")
                    summary_file_tsl=open(common.path_summaries_tsl+"/"+fn, "w")
                    text_file_tslr=open(common.path_texts_tslr+"/"+fn, "w")
                    summary_file_tslr=open(common.path_summaries_tslr+"/"+fn, "w")
                    
                    sign=False
                    for line in text_file_ts.readlines():
                        line=line.strip()
                        if sign:
                            text_file_tsl.write("\n")
                            text_file_tslr.write(" ")
                        text_file_tsl.write(common.preprocess_line(line))
                        text_file_tslr.write(common.preprocess_line(line))
                        sign=True
                    
                    sign=False
                    for line in summary_file_ts.readlines():
                        line=line.strip()
                        if sign:
                            summary_file_tsl.write("\n")
                            summary_file_tslr.write(" ")
                        summary_file_tsl.write(common.preprocess_line(line))
                        summary_file_tslr.write(common.preprocess_line(line))
                        sign=True
                    
                    summary_file_ts.close()
                    text_file_ts.close()
                    summary_file_tsl.close()
                    text_file_tsl.close()
                    summary_file_tslr.close()
                    text_file_tslr.close()
                    
    def create_mentions_ex():
        count=0
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    mention_me_file=open(common.path_bekerlay_text_mentions+"/"+fn) 
                    mention_lines=mention_me_file.readlines()
                    mention_me_file.close()
                    
                    mention_ex_file=open(common.path_mentions_ex+"/"+fn,"w") 
                    
                    entity_names={}
                    for mention_line in mention_lines:
                        mention_line=mention_line.strip()
                        items=mention_line.split("\t")
                        if items[0] not in entity_names:
                            entity_names[items[0]]=items[1]
                        elif len(items[1].split())>len(entity_names[items[0]].split()):
                            entity_names[items[0]]=items[1]
                        
                    mentions=[]
                    for mention_line in mention_lines:
                        mention_line=mention_line.strip()
                        items=mention_line.split("\t")
                        mention_ex_file.write("%s\t%s\t%s\t%s\t%s\n" % (items[0], entity_names[items[0]], items[2], items[3], items[4]))
                    
                    mention_ex_file.close()
                    count+=1
                    print("mention_uniform: %d" % (count))
    
    def entities2mentions():
        count=0
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    mention_me_file=open(common.path_bekerlay_text_mentions+"/"+fn) 
                    mention_lines=mention_me_file.readlines()
                    mention_me_file.close()
                    
                    e2m_file=open(common.path_entities2mentions+"/"+fn,"w") 
                    
                    entities=[]
                    e2m={}
                    for mention_line in mention_lines:
                        mention_line=mention_line.strip()
                        items=mention_line.split("\t")
                        if items[0] not in e2m:
                            e2m[items[0]]=[]
                            entities.append(items[0])
                        mentions=e2m[items[0]]
                        mentions.append(items[1])
                        
                    for entity in entities:
                        e2m_file.write(entity+"\t"+"\t".join(e2m[entity])+"\n")
                    
                    e2m_file.close()
                    count+=1
                    print("e2m: %d" % (count))
                    
    def entities2mentions_labeled():
        count=trues=0
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    
                    summary_file=open(common.path_summaries_ts+"/"+fn) 
                    text_lines=summary_file.readlines()
                    text_lines=[text_line.strip() for text_line in text_lines]
                    summary_file.close()
                    
                    e2m_file=open(common.path_entities2mentions+"/"+fn) 
                    e2m_labeled_file=open(common.path_entities2mentions_labeled+"/"+fn,"w") 
                    
                    entities=[]
                    for line in e2m_file.readlines():
                        items=line.strip().split("\t")
                        tmp={}
                        true_false=0
                        entity_id=items[0]
                        entities.append(entity_id)
                        item_count=len(items)
                        i=1
                        while(i<item_count) and not true_false:
                            item=items[i]
                            if item not in tmp:
                                tmp[item]=1
                                for text_line in text_lines:
                                    if text_line.find(item)>-1:
                                        true_false=1
                                        trues+=1
                                        break
                            i+=1
                        entities.append((entity_id, true_false))
                        e2m_labeled_file.write(line.strip()+"\t"+str(true_false)+"\n")
                    
                    e2m_labeled_file.close()
                    e2m_file.close()
                    
                    count+=1
                    print("e2m-label: %d" % (count))
        print(count, trues/count)
        
    def entities2mentions_labeled_2():
        #104282 2.2233750791124067
        count=0
        avg_trues=3
        fns=[]
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    e2m_labeled_file=open(common.path_entities2mentions_labeled+"/"+fn) 
                    
                    trues=0
                    for line in e2m_labeled_file.readlines():
                        #print(line.strip())
                        items=line.strip().split("\t")
                        trues+=int(items[-1])
                    if trues==0:
                        fns.append(fn)
                    
                    e2m_labeled_file.close()
        
        print(len(fns))
        print(fns[0])
        
        for fn in fns:
            e2m_file=open(common.path_entities2mentions+"/"+fn) 
            e2m_lines=e2m_file.readlines()
            e2m_file.close()
            
            entities=[]
            for line in e2m_lines:
                items=line.strip().split("\t")
                entities.append(items[:-1])
            entities.sort(key=lambda x:(len(x)))
            
            elabels={}
            for entity in entities[:avg_trues]:
                elabels[entity[0]]=1
                
            e2m_labeled_file=open(common.path_entities2mentions_labeled+"/"+fn, "w") 
            for line in e2m_lines:
                items=line.strip().split("\t")
                if items[0] in elabels:
                    e2m_labeled_file.write(line.strip()+"\t1\n")
                else:
                    e2m_labeled_file.write(line.strip()+"\t0\n")
            e2m_labeled_file.close()
    
    #create_stories_texts_summaries_ts()
    #tslr()
    #create_mentions_ex()
    #entities2mentions()
    #entities2mentions_labeled()
    entities2mentions_labeled_2()

def process_entity_embeddings():
    
    fn_file_paths=[common.path_train_file, common.path_dev_file, common.path_test_file]
    
    def entity_linking():
        #run codes in Java
        pass
    
    def create_entity_vocab():
        #146894
        entity_counts={}
        for fn_file_path in fn_file_paths:
            with open(fn_file_path) as fn_file:
                for fn in fn_file.readlines():
                    fn=fn.strip()
                    try:
                        file_entities = open(common.path_entities+"/"+data_name)
                        for line in file_entities:
                            line=line.strip()
                            items=line.split("\t")
                            if not items[2] in entity_counts:
                                entity_counts[items[2]]=0
                            entity_counts[items[2]]+=1
                        file_entities.close()
                    except: print(data_name)
        
        file_entity_vocab=open(common.path_entity_vocab, "w")
        entities=list(entity_counts.keys())
        entities.sort(key=lambda x:entity_counts[x], reverse=True)
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
    
    #entity_linking()
    #create_entity_vocab()
    create_entity_embedding()
    #test()


def remove_ner_error():
    ner_errors=[]
    with open(path_corpus+"/notexist") as file:
        for line in file.readlines():
            ner_errors.append(line.strip())
    
    paths=[path_train_file, path_dev_file, path_dev_file]
    for path1 in paths:
        stories=[]
        with open(path1) as file:
            for line in file.readlines():
                line=line.strip()
                if line not in ner_errors:
                    stories.append(line)
        
        with open(path1, "w") as w_file:
            for file_name in stories:
                w_file.write(file_name+"\n")


def process_word_embedding():
    import word2vec
    
    fn_file_paths=[common.path_train_file, common.path_dev_file, common.path_test_file]
    
    def merge_text():
        count=0
        w_file = open("/home/test/kbsumm/data/glove/wiki-dm-cnn-nyt50.txt", "a")
        w_file.write(" ")
        file_dirs=[common.path_texts_tslr, common.path_summaries_tslr]
        for file_dir in file_dirs:
            fns=os.listdir(file_dir)
            for fn in fns:
                file = open(file_dir+"/"+fn)
                text=file.read().strip()
                w_file.write(" "+text)
                file.close()
                count+=1
                print("nyt50:",count)
        
        w_file.close()
    
    def build_word_vocab():
        data=[]
        file_dirs=[common.path_texts_tslr, common.path_summaries_tslr]
        for file_dir in file_dirs:
            fns=os.listdir(file_dir)
            for fn in fns:
                with open(file_dir+'/'+fn) as file:
                    text=file.read().replace('\n','')
                    words=text.split()
                    data.extend(words)
    
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        word_list=[word for word, count in count_pairs]
        
        count=0 
        with open(common.path_word_vocab, 'w') as vocab:
            for item1, item2 in count_pairs:
                if count==40000:
                    break
                print(item1, item2)
                vocab.write('%s\t%s\n' % (item1, item2))
                count=count+1
    
    def build_total_embeddings():
        word2vec.word2vec('/home/test/kbsumm/data/glove/wiki-dm-cnn-nyt50.txt', '/home/test/kbsumm/data/glove/wiki-dm-cnn-nyt50-word2vec-128.bin', size=128, verbose=True)
        
    def word_vocab_embedding():
        model = word2vec.load('/home/test/kbsumm/data/glove/wiki-dm-cnn-nyt50-word2vec-128.bin')
        with open(common.path_word_vocab) as file:
            vocab=[line.split('\t')[0] for line in file.readlines()]
            embeddings=[]
            count=0
            for word in vocab:
                embeddings.append(model[word])
            embeddings=np.array(embeddings,np.float32)
            embeddings.tofile(common.path_word_vocab_embeddings_word2vec)
    
    #tokenize_wiki_text2()
    #merge_text()
    build_word_vocab()
    build_total_embeddings()
    word_vocab_embedding()

def build_ext_gold_standard():
    import rouge
    import struct
    
    evaluator = rouge.Rouge(metrics=['rouge-n', "rouge-l"],
                               max_n=2,
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=True)
    
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
                tmp_summary_score=rouge_score(evaluator.get_scores(tmp_summary,ref_summary))
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
        #if os.path.exists(common.path_summaries_ext_tsl+'/'+file_name):
        #    return False
        
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
    
    def find_error_files():
        file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
        error_file=open(common.path_corpus+"/build_ext_gold_standard-error", "w")
        
        for file_name in file_names:
            with open(file_name) as file:
                for line in file.readlines():
                    line=line.strip()
                    if not os.path.exists(common.path_summaries_ext_tsl+'/'+line):
                        error_file.write(line+"\n")
                        error_file.flush()
        
        error_file.close()
    
    #Build()
    find_error_files()


def build_ext_gold_standard_2():
    import rouge
    import struct
    evaluator = rouge.Rouge(metrics=['rouge-n', "rouge-l"],
                               max_n=2,
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               apply_avg=False,
                               stemming=True)
    error_file=open(common.path_corpus+"/build_ext_gold_standard_2-error", "w")

    def greedy(text_lines, ext_indices, ref_summary, file_name):
        summary_lines=[]
        
        for i in range(len(text_lines)):
            line=text_lines[i].strip("\n").strip(" ")
            if line=='':
                continue
            
            if ext_indices[i]==0:
                scores=evaluator.get_scores([line] * len(ref_summary),ref_summary)["rouge-l"]
                scores.sort(key=lambda x:x['f'][0], reverse=True)
                score=scores[0]['f'][0]
                if score>=0.5:
                    summary_lines.append(line)
                    ext_indices[i]=1
                    
    def Mapper(t):
        file_name,n=t
        if os.path.exists(common.path_summaries_ext_tsl_2+'/'+file_name):
            return False
        
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
            len1=np.sum(ext_indices)
            #print(n, file_name, len(text_lines), len(ext_indices))
            #print(ext_indices)
            #print(n)
            greedy(text_lines,ext_indices,ref_summary,file_name)
        except:
            error_file.write(file_name+"\n")
            return
        
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
        #print(n, file_name,'-----------------------------%s----------------------------' % file_name)
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
            
    def find_error_files():
        file_names=[common.path_train_file, common.path_dev_file, common.path_test_file]
        error_file=open(common.path_corpus+"/build_ext_gold_standard_2-error", "w")
        
        for file_name in file_names:
            with open(file_name) as file:
                for line in file.readlines():
                    line=line.strip()
                    if not os.path.exists(common.path_summaries_label_tsl_2+'/'+line):
                        error_file.write(line+"\n")
                        error_file.flush()
        
        error_file.close()
                    
    Build()
    #find_error_files()
    
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

def max_sent_entity_num():
    max_sent_num=0
    avg_sent_num=0
    max_true_sent_num=0
    avg_true_sent_num=0
    count=0
    with open(common.path_corpus+"/trains (3rd copy)") as file:
        for fn in file.readlines():
            fn=fn.strip()
            count+=1
            
            summary_labels=np.fromfile(common.path_summaries_label_tsl_2+"/"+fn, np.float32)
            t_sent_num=len(summary_labels)
            avg_sent_num += t_sent_num
            if max_sent_num < t_sent_num:
                max_sent_num = t_sent_num
            
            t_true_sent_num=0
            for i in summary_labels:
                if i==1: t_true_sent_num+=1
            
            avg_true_sent_num += t_true_sent_num
            if max_true_sent_num < t_true_sent_num:
                max_true_sent_num = t_true_sent_num
            
    print("max_sent_num", max_sent_num)
    print("avg_sent_num", avg_sent_num/count)
    print("max_true_sent_num", max_true_sent_num)
    print("avg_true_sent_num", avg_true_sent_num/count)
    
    avg_true_entities=0
    max_true_entities=0
    avg_entities=0
    max_entities=0
    with open(common.path_corpus+"/trains (3rd copy)") as file:
        for file_line in file.readlines():
            file_line = file_line.strip()
            with open(common.path_entities2mentions_labeled+"/"+file_line) as file:
                lines=file.readlines()
            
            t_true_entities = 0
            for line in lines:
                line=line.strip()
                items=line.split("|||")
                if items[-1]=="1": t_true_entities+=1
            if max_true_entities < t_true_entities:
                max_true_entities = t_true_entities
            avg_true_entities+=t_true_entities
            avg_entities += len(lines)
            if max_entities < len(lines):
                max_entities = len(lines)
    print("max_true_entities", max_true_entities)
    print("avg_true_entities", avg_true_entities/count)
    print("avg_entities", avg_entities/count)
    print("max_entities", max_entities)
                

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
                            items=line.split("\t")
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
    
def test01():
    lines = []
    with open(path_xmls+"/1165027.xml") as file:
        for line in file.readlines():
            lines.append(line.strip())
        doc=NYTDoc(lines)
    
    if "abstract" in doc.summaries and hasattr(doc, "full_text") and doc.headlines!={}:
        print("[HEADLINE]")
        if "print" in doc.headlines:
            print(doc.headlines["print"])
        else:
            print(doc.headlines["online"])
            
        print("\n[BYLINE]")
        if "byline" in doc.summaries:
            for line in doc.summaries["byline"]:
                print(line)
            
        print("\n[ABSTRACT]")
        for line in doc.summaries["abstract"]:
            print(line)
            
        print("\n[FULL_TEXT]")
        for line in doc.full_text:
            print(line)
    else:
        #if "abstract" not in doc.summaries:print("abstract")
        #if not hasattr(doc, "full_text"):print("full_text")
        #if "headline" not in doc.summaries:print("headline")
        print(1)
        pass

def test02():
    file_paths=[path_all_file, path_valid_file, path_error_file]
    for file_path in file_paths:
        store=[]
        with open(file_path) as file:
            for line in file.readlines():
                store.append(line.strip().split(".")[0])
        with open(file_path, "w") as w_file:
            for line in store:
                w_file.write(line+"\n")

def test03():
    #英文中的应用
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(r'/home/test/eclipse/libs/stanford-corenlp-full-2018-10-05')
    
    sentence = 'Guangdong University of Foreign Studies is located in Guangzhou. Guangdong University of Foreign Studies is located in Guangzhou.'
    print ('Tokenize:', nlp.word_tokenize(sentence))
    
    print ('Part of Speech:', nlp.pos_tag(sentence))
    print ('Named Entities:', nlp.ner(sentence))
    print ('Constituency Parsing:', nlp.parse(sentence))#语法树
    print ('Dependency Parsing:', nlp.dependency_parse(sentence))#依存句法nlp.close() # Do not forget to close! The backend server will consume a lot memery

def test04():
    import rouge
    from nltk.tokenize import sent_tokenize
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                   max_n=2,
                                   limit_length=True,
                                   length_limit=100,
                                   length_limit_type='words',
                                   alpha=0.5, # Default F1_score
                                   weight_factor=1.2,
                                   stemming=True)
    
    tfile = open(common.path_train_file)
    ref=[]
    ext=[]
    for fn in tfile.readlines():
        fn=fn.strip()
        
        with open(common.path_summaries_tsl+"/"+fn) as file:
            ref.append(file.read().strip())
        
        with open(common.path_summaries_ext_tsl_2+"/"+fn) as file:
            ext.append(file.read().strip())
            
    scores_ext = evaluator.get_scores(ext, ref)
    print("%s\t%s\t%s\n" % (scores_ext["rouge-1"]["f"], scores_ext["rouge-2"]["f"], scores_ext["rouge-l"]["f"]))

    tfile.close()

def main():
    #berkeley_sents()
    #berkeley_mentions()
    #create_file_list()
    #process_stories()
    #process_entity_embeddings()
    #remove_ner_error()
    #process_word_embedding()
    #build_ext_gold_standard()
    #build_ext_gold_standard_2()
   # entity_gold_standard()
    #max_sent_entity_num()
    statistics()
    #test03()
    #test04()

if __name__ == '__main__':
    main()





