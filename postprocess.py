import common
import os, re
import argparse
import numpy as np

path_results=common.path_corpus+"/results-kb"

models=["W.O.RL-gcn0", "W.O.RL-gcn1", "W.O.RL-gcn2", "W.O.RL-gcn3"]
models=["TEO-10"] 

def accumulate_ext_gen(corpus="dailymail", model_base=common.path_log_root_gcn):
    
    def accumulate_one(dir_name):
        file_names=os.listdir(model_base+"/"+dir_name+"/ext_greedy_decode-dailymail")
        file_names=sorted(file_names)
        #file_names=file_names[::-1]
        
        out_dir=path_results+"/accumulate/" + corpus
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        w_file=open(out_dir+"/"+dir_name, "w")
        for file_name in file_names:
            
            r_file=open(model_base + "/" + dir_name + "/ext_greedy_decode-" + corpus + "/" + file_name)
            lines=r_file.readlines()
            lines=lines[::-1]
            print(dir_name+"/"+file_name)
            
            strs=[]
            for i in range(len(lines)):
                line=lines[i].strip("\n")
                if re.match("^\[(\d+)\]$",line):
                    break
                
                strs.insert(0, line)
            
            w_file.write("%s-%s-%s\t%s-%s-%s\t%s-%s-%s\n" % (strs[12].split("\t")[-1], strs[14].split("\t")[-1], strs[16].split("\t")[-1],
                                                             strs[22].split("\t")[-1], strs[24].split("\t")[-1], strs[26].split("\t")[-1],
                                                             strs[32].split("\t")[-1], strs[34].split("\t")[-1], strs[36].split("\t")[-1]))
            
            r_file.close()
        w_file.close()
        
    def accumulate_two():
        file_names=os.listdir(path_results+"/accumulate/" + corpus)
        file_names.sort()
        w_file_bleu=open(path_results+"/"+corpus+".txt", "w")
        for file_name in file_names:
            r_file=open(path_results+"/accumulate/"+corpus+"/"+file_name)
            lines=r_file.readlines()
            max_i=0
            max_value=-1
            for i in range(len(lines)):
                line=lines[i].strip("\n")
                print(file_name+":"+line)
                value_bleu=float(line.split("|||")[0].split("|")[1].strip(" ").split(" ")[0])
                if value_bleu>max_value:
                    max_value=value_bleu
                    max_i=i
            w_file_bleu.write(file_name+"-"+lines[max_i])
            r_file.close()
        w_file_bleu.close()
    
    
    for file_dir in models:
        accumulate_one(file_dir)
    
    #accumulate_two()


def accumulate_gen(corpus="cnn+dm", model_base=common.path_log_root_gcn):
    models=["TEO-10-noen", "TEO-4-noen"] 
    models=["TEO-4-noen", "TEO-4-noen.0921"] 
    def accumulate_one(dir_name):
        file_names=os.listdir(model_base+"/"+dir_name+"/greedy-dailymail")
        file_names=sorted(file_names)
        #file_names=file_names[::-1]
        
        out_dir=path_results+"/accumulate/" + corpus
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        w_file=open(out_dir+"/"+dir_name, "w")
        for file_name in file_names:
            
            r_file=open(model_base + "/" + dir_name + "/greedy-dailymail" + "/" + file_name)
            lines=r_file.readlines()
            lines=lines[::-1]
            print(dir_name+"/"+file_name)
            
            strs=[]
            for i in range(len(lines)):
                line=lines[i].strip("\n")
                if re.match("^\[(\d+)\]$",line):
                    break
                
                strs.insert(0, line)
            
            w_file.write("%s\t%s\t%s\t%s\n" % (file_name, strs[6].split("\t")[-1], strs[8].split("\t")[-1], strs[10].split("\t")[-1]))
            
            r_file.close()
        w_file.close()
        
    def accumulate_two():
        file_names=os.listdir(path_results+"/accumulate/" + corpus)
        file_names.sort()
        w_file_bleu=open(path_results+"/"+corpus+".txt", "w")
        for file_name in file_names:
            r_file=open(path_results+"/accumulate/"+corpus+"/"+file_name)
            lines=r_file.readlines()
            max_i=0
            max_value=-1
            for i in range(len(lines)):
                line=lines[i].strip("\n")
                print(file_name+":"+line)
                value_bleu=float(line.split("|||")[0].split("|")[1].strip(" ").split(" ")[0])
                if value_bleu>max_value:
                    max_value=value_bleu
                    max_i=i
            w_file_bleu.write(file_name+"-"+lines[max_i])
            r_file.close()
        w_file_bleu.close()
    
    
    for file_dir in models:
        accumulate_one(file_dir)
    
    #accumulate_two()
    

def accumulate_train(corpus="dailymail", model_base=common.path_log_root_gcn):
    
    models=["W.O.RL-10-gcn2-0.00", "W.O.RL-10-gcn3-0.00", "W.O.RL-4-gcn2-0.00", "W.O.RL-4-gcn3-0.00"] 
    models=["W.O.RL-4-gcn2-0.00-noen", "W.O.RL-4-gcn2-0.00"] 
    models=["W.O.RL-4-gcn2-0.00-ee0.1", "W.O.RL-4-gcn2-0.00-noen"] 
    models=["W.O.RL-10-gcn2-0.00-ee0.1"] 
    
    def accumulate_one(dir_name):
        file_names=os.listdir(model_base+"/"+dir_name+"/train-ext-gen-dailymail")
        file_names=sorted(file_names)
        #file_names=file_names[::-1]
        
        out_dir=path_results+"/accumulate/" + corpus
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        w_file=open(out_dir+"/"+dir_name, "w")
        for file_name in file_names:
            
            r_file=open(model_base + "/" + dir_name + "/train-ext-gen-dailymail/" + file_name)
            lines=r_file.readlines()
            print(dir_name+"/"+file_name)
            
            ave = [0.] * 6
            for i in range(len(lines)):
                line = lines[i]
                items = line.split("\t")
                items = [float(item) for item in items]
                
                for j in range(6):
                    ave[j]+=items[j]
                
            for j in range(6):
                ave[j]/=len(lines)
            
            w_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (file_name, ave[0], ave[1], ave[2], ave[3], ave[4], ave[5]))
            
            r_file.close()
        w_file.close()
    
    
    for file_dir in models:
        accumulate_one(file_dir)
    
    #accumulate_two()
    
def accumulate_train2(model_base=common.path_log_root_gcn):
    import rouge
    from nltk.tokenize import sent_tokenize
    
    models=["W.O.RL-4-gcn0-0.00-ee0.1"] 
    models=["W.O.RL-4-gcn2-0.00-ee0.1"] 
    models=["W.O.RL-4-gcn2-0.00-ee0.1-noge"] 
    models=["W.O.RL-4-gcn2-0.00-ee0.1-1007"] 
    models=["W.O.RL-20-gcn2-0.00-ee0.1"]
    models=["W.O.RL-20-gcn2-0.00-noen"]
    models=["W.O.RL-50-gcn2-0.00-ee0.1"]
    models=["W.O.RL-50-gcn2-0.00-ee0.1-old"]
    models=["W.O.RL-50-gcn0-0.00-ee0.1"]
    models=["W.O.RL-50-gcn2-0.30-ee0.1"]
    models=["W.O.RL-50-gcn2-0.00-noen-ee0.1-noge"]
    models=["W.O.RL-50-gcn2-0.00-noen"]
    models=["W.O.RL-50-gcn2-0.00-noen-sota", "W.O.RL-50-gcn2-0.00-ee0.1-sota"]
    models=["W.O.RL-50-gcn2-0.00"] 
    models=["W.O.RL-50-gcn2-0.00-noge"] 
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                   max_n=2,
                                   limit_length=True,
                                   length_limit=100,
                                   length_limit_type='words',
                                   alpha=0.5, # Default F1_score
                                   weight_factor=1.2,
                                   stemming=True)
    
    def accumulate_one(dir_name):
        file_names=os.listdir(model_base+"/"+dir_name+"/train-ext-gen-dailymail")
        file_names=sorted(file_names)
        out_dir=path_results+"/accumulate/cnn+dm"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        w_file=open(out_dir+"/"+dir_name, "w")
        #print(file_names)
        for file_name in file_names:
            lines=[]
            gen=[]
            ext=[]
            ref=[]
            with open(model_base + "/" + dir_name + "/train-ext-gen-dailymail/" + file_name) as file:
                for line in file.readlines():
                    lines.append(line.strip())
            for i in xrange(len(lines)//4 * 4):
                if i%4==0:
                    sents=sent_tokenize(lines[i])
                    gen.append("\n".join(sents))
                elif i%4==1:
                    sents=sent_tokenize(lines[i])
                    ext.append("\n".join(sents))
                elif i%4==2:
                    sents=lines[i].split("|||")
                    ref.append("\n".join(sents))
            
            scores_gen = evaluator.get_scores(gen, ref)
            scores_ext = evaluator.get_scores(ext, ref)
            
            w_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (file_name, scores_gen["rouge-1"]["f"], scores_gen["rouge-2"]["f"], scores_gen["rouge-l"]["f"], 
                                                                      scores_ext["rouge-1"]["f"], scores_ext["rouge-2"]["f"], scores_ext["rouge-l"]["f"]))
            w_file.flush()
            print(file_name)
        w_file.close()
        
    for file_dir in models:
        accumulate_one(file_dir)
        
def evaluate_ext_gen(model_base=common.path_log_root_gcn):
    import rouge
    from nltk.tokenize import sent_tokenize
    
    models=[("W.O.RL-50-gcn2-0.00-ee0.1", ["553827", "423683"])]#423683
    #models=[("W.O.RL-50-gcn0-0.00-ee0.1", ["529855"])] #597074
    #models=[("W.O.RL-50-gcn2-0.30-ee0.1", ["430569"])]
    #models=[("W.O.RL-50-gcn2-0.00-noen-ee0.1-noge", ["506169"])]#932818
    #models=[("W.O.RL-50-gcn2-0.00-noen", ["516337"])]#447744
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                   max_n=2,
                                   limit_length=True,
                                   length_limit=100,
                                   length_limit_type='words',
                                   alpha=0.5, # Default F1_score
                                   weight_factor=1.2,
                                   stemming=True)
    
    def accumulate_one(model_name, ckpt, w_file):
        
        #print(file_names)
        lines=[]
        gen=[]
        ref=[]
        with open(model_base + "/" + model_name + "/ext-gen-dailymail/" + ckpt+".ge") as file:
            for line in file.readlines():
                lines.append(line.strip())
            for i in xrange(len(lines)//2 * 2):
                if i%2==0:
                    sents=sent_tokenize(lines[i])
                    gen.append("\n".join(sents))
                elif i%2==1:
                    sents=lines[i].split("|||")
                    ref.append("\n".join(sents))
        
        lines=[]       
        ext=[]
        with open(model_base + "/" + model_name + "/ext-gen-dailymail/" + ckpt+".ex") as file:
            for line in file.readlines():
                sents=sent_tokenize(line.strip())
                ext.append("\n".join(sents))
            
        scores_gen = evaluator.get_scores(gen, ref)
        scores_ext = evaluator.get_scores(ext, ref)
            
        w_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (ckpt, scores_gen["rouge-1"]["f"], scores_gen["rouge-2"]["f"], scores_gen["rouge-l"]["f"], 
                                                                      scores_ext["rouge-1"]["f"], scores_ext["rouge-2"]["f"], scores_ext["rouge-l"]["f"]))
        w_file.flush()
        print(ckpt)
    
    out_dir=path_results+"/evaluation/cnn+dm"   
    if not os.path.exists(out_dir):
            os.mkdir(out_dir) 
    
    for file_dir in models:
        w_file=open(out_dir+"/"+file_dir[0], "w")
        for ckp in file_dir[1]:
            accumulate_one(file_dir[0], ckp, w_file)
        w_file.close()

if __name__ == '__main__':
    #accumulate_ext_gen("dailymail")
    #accumulate_gen(corpus="cnn+dm", model_base=common.path_log_root)
    accumulate_train2(model_base=common.path_log_root_gcn)
    #evaluate_ext_gen(model_base=common.path_log_root_gcn)





