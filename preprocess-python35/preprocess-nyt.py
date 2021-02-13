# -*- coding: utf-8
import os
import common
from NYTDoc import NYTDoc 

path_corpus = common.path_base + "/data/nyt"
path_corpus_orgin = path_corpus + "/nyt_corpus/data"
path_xmls = path_corpus + "/xmls"

path_all_file = path_corpus+"/all"
path_valid_file = path_corpus+"/valid"
path_error_file = path_corpus+"/error"
path_train_file = path_corpus+"/trains"
path_dev_file = path_corpus+"/devs"
path_test_file = path_corpus+"/tests"

path_stories = path_corpus+"/stories"
path_texts = path_corpus+"/texts"
path_summaries = path_corpus +"/summaries"
path_entities = path_corpus + "/entities"

def move_xmls():
    if not os.path.exists(path_xmls):os.mkdir(path_xmls)
    
    dir_years = os.listdir(path_corpus_orgin)
    for year in dir_years:
        if int(year)<2001:continue
        dir_months = os.listdir(path_corpus_orgin+"/"+year)
        for month in dir_months:
            if not os.path.isdir(path_corpus_orgin+"/"+year+"/"+month):
                continue
            dirs = os.listdir(path_corpus_orgin+"/"+year+"/"+month)
            for dir in dirs:
                file_names = os.listdir(path_corpus_orgin+"/"+year+"/"+month+"/"+dir)
                for file_name in file_names:
                    os.system("mv %s %s" % (path_corpus_orgin+"/"+year+"/"+month+"/"+dir+"/"+file_name, path_xmls+"/"+file_name))

def extract_stories_from_xmls():
    count=0
    count1=0
    count2=0
    file_names = os.listdir(path_xmls)
    file_names = sorted(file_names)
    all_file = open(path_all_file, "w")
    valid_file = open(path_valid_file, "w")
    error_file = open(path_error_file, "w")
    
    if not os.path.exists(path_stories):os.mkdir(path_stories)
    
    for file_name in file_names:
        all_file.write(file_name+"\n")
        count+=1
        lines=[]
        #with open(path_xmls+"/1165027.xml") as file:
        with open(path_xmls+"/"+file_name, encoding ='utf-8') as file:
            #print(path_xmls+"/"+file_name)
            for line in file.readlines():
                lines.append(line.strip())
        doc=NYTDoc(lines)
        if "abstract" in doc.summaries and hasattr(doc, "full_text") and doc.headlines!={}:
            w_file = open(path_stories+"/"+file_name.split(".")[0], "w")
            w_file.write("[HEADLINE]\n")
            if "print" in doc.headlines:
                w_file.write(doc.headlines["print"]+"\n")
            else:
                w_file.write(doc.headlines["online"]+"\n")
            
            w_file.write("\n[BYLINE]\n")
            if "byline" in doc.summaries:
                for line in doc.summaries["byline"]:
                    w_file.write(line+"\n")
            
            w_file.write("\n[ABSTRACT]\n")
            for line in doc.summaries["abstract"]:
                w_file.write(line+"\n")
            
            w_file.write("\n[FULL_TEXT]\n")
            for line in doc.full_text:
                w_file.write(line+"\n")
            
            w_file.close()
            
            valid_file.write(file_name+"\n")
            
            count1+=1
            pass
        else:
            #if "abstract" not in doc.summaries:print("abstract")
            #if not hasattr(doc, "full_text"):print("full_text")
            #if "headline" not in doc.summaries:print("headline")
            error_file.write(file_name+"\n")
            count2+=1
        if count % 10000 == 0:print(count, count1, count2)
    all_file.close()
    valid_file.close()
    error_file.close()
    print(count, count1, count2)
            
def split_90_5_5():
    store=[]
    with open(path_valid_file) as file:
        for line in file.readlines():
            store.append(line.strip())
    length=len(store)
    len_train=round(length*0.9)
    len_dev=(length-len_train)//2
    len_test=length-len_train-len_dev
    
    with open(path_train_file, "w") as w_file:
        for file_name in store[:len_train]:
            w_file.write(file_name+"\n")
            
    with open(path_dev_file, "w") as w_file:
        for file_name in store[len_train:len_train+len_dev]:
            w_file.write(file_name+"\n")
            
    with open(path_test_file, "w") as w_file:
        for file_name in store[len_train+len_dev:length]:
            w_file.write(file_name+"\n")

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

def process_entity_embeddings():
    
    file_names=[path_train_file, path_dev_file, path_test_file]
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
        entities=entities[:100000]
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

def count():
    files = os.listdir(path_xmls)
    c=0
    for file in files:
        if file.find("copy")>=0:print(1,file)
        print(file)
        c+=1
    print(c)
    
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

def main():
    #move_xmls()
    #count()
    #extract_stories_from_xmls()
    #split_90_5_5()
    remove_ner_error()
    #test03()

if __name__ == '__main__':
    main()





