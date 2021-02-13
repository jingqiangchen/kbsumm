import common
import os, re
import argparse

def train(mode="TEO", coverage="False", batch_size=10, gpu=1, corpus="dailymail", en_vocab_size=4, use_entity_embedding="True"):
    
    if gpu is None:
        gpu=1
    
    pointer_gen="True"
    coverage="True"
    
    if mode=="TEO":
        pass
    if mode=="TM":
        use_entity_embedding=False
    
    if en_vocab_size==5:
        entity_vocab_path=common.path_entity_vocab_5
        entity_embedding_path=common.path_entity_vocab_embeddings_5
        exp_name=mode+"-5"
    elif en_vocab_size==10:
        entity_vocab_path=common.path_entity_vocab_10
        entity_embedding_path=common.path_entity_vocab_embeddings_10
        exp_name=mode+"-10"
    elif en_vocab_size==20:
        entity_vocab_path=common.path_entity_vocab_20
        entity_embedding_path=common.path_entity_vocab_embeddings_20
        exp_name=mode+"-20"
    
    if use_entity_embedding=="False":
        exp_name+="-noen"
    
    entity_vocab_size=en_vocab_size*10000 + 4
    data_path=common.path_chunked_train+"/*"
    #data_path="/home/test/kbsumm/data/dailymail/chunked/example.bin"
 
    os.system('''
        export CUDA_VISIBLE_DEVICES=%d
            
        python -m run_summarization \
              --data_path="%s" \
              --exp_name=%s \
              --batch_size=%d \
              --pointer_gen=%s \
              --coverage=%s \
              --use_entity_embedding=%s \
              --entity_vocab_path="%s" \
              --entity_embedding_path="%s" \
              --entity_vocab_size=%d \
              --corpus=%s 
    ''' % (gpu, data_path, exp_name, batch_size, pointer_gen, coverage, use_entity_embedding, 
           entity_vocab_path, entity_embedding_path, entity_vocab_size, corpus))
    

def beam_search(mode="TEO", beam_size=5, gpu=1, corpus="dailymail", is_batch=True, en_vocab_size=4, dec_method="beam-search", use_entity_embedding="True"):
    
    if gpu is None:
        gpu=1
      
    common.path_chunked_train=common.path_chunked+"/train"
    common.path_chunked_dev=common.path_chunked+"/dev"
    common.path_chunked_test=common.path_chunked+"/test"
        
    exp_name = mode
    pointer_gen="True"
    coverage="True"
    max_enc_steps=150
        
    if en_vocab_size==4:
        entity_vocab_path=common.path_entity_vocab
        entity_embedding_path=common.path_entity_vocab_embeddings
        exp_name=mode+"-4"
    elif en_vocab_size==10:
        entity_vocab_path=common.path_entity_vocab_10
        entity_embedding_path=common.path_entity_vocab_embeddings_10
        exp_name=mode+"-10"
    elif en_vocab_size==20:
        entity_vocab_path=common.path_entity_vocab_20
        entity_embedding_path=common.path_entity_vocab_embeddings_20
        exp_name=mode+"-20"
        
    if use_entity_embedding=="False":
        exp_name+="-noen"
    
    data_path=common.path_chunked_dev+"/*"
    entity_vocab_size=en_vocab_size*10000 + 4
    path_ckpt = common.path_log_root + "/" + exp_name + "/train/checkpoint"
    
    if is_batch:
        os.system("cp '%s' '%s'" % (path_ckpt, path_ckpt+"-bak"))
    
    min_ckpt=100000
    max_ckpt=800000
    cur_ckpt=100000
    new_first_line=""
    sign=True
    while True and sign: 
        sign=False
        if is_batch and cur_ckpt>=min_ckpt and cur_ckpt<=max_ckpt :
            with open(path_ckpt) as file:
                lines=file.readlines()
                lines=lines[1:]
                
                for line in lines:
                    line=line.strip("\n").strip("\"")
                    items=line.split("-")
                    if int(items[-1])>=cur_ckpt:
                        cur_ckpt=int(items[-1])+1
                        new_first_line="model_checkpoint_path:"+line.split(":")[-1]+"\""
                        sign=True
                        print(line)
                        break
            
            if sign:
                with open(path_ckpt, "w") as file:
                    file.write(new_first_line+"\n")
                    for line in lines:
                        line=line.strip("\n")
                        file.write(line+"\n")
        
        os.system('''
                export CUDA_VISIBLE_DEVICES=%d
                
                python -m run_summarization \
                      --mode=decode \
                      --single_pass=True \
                      --data_path="%s" \
                      --exp_name=%s \
                      --pointer_gen=%s \
                      --beam_size=%d \
                      --coverage=%s \
                      --use_entity_embedding=%s \
                      --max_enc_steps=%d \
                      --entity_vocab_path="%s" \
                      --entity_embedding_path="%s" \
                      --dec_method=%s \
                      --entity_vocab_size=%d \
                      --corpus=%s 
            ''' % (gpu, data_path, exp_name, pointer_gen, beam_size, coverage, use_entity_embedding, 
                   max_enc_steps, entity_vocab_path, entity_embedding_path, dec_method, entity_vocab_size, corpus))
        
        if not is_batch:
            break
        

def eval(mode="TEO", beam_size=5, gpu=1, corpus="dailymail", en_vocab_size=4):
    
    if gpu is None:
        gpu=1
      
    common.path_chunked_train=common.path_chunked+"/train"
    common.path_chunked_dev=common.path_chunked+"/dev"
    common.path_chunked_test=common.path_chunked+"/test"
        
    exp_name = mode
    pointer_gen="True"
    coverage="True"
    use_entity_embedding="True"
    max_enc_steps=150
    
    if mode=="TEO":
        pass
    if mode=="TM":
        use_entity_embedding=False
        
    if en_vocab_size==4:
        entity_vocab_path=common.path_entity_vocab
        entity_embedding_path=common.path_entity_vocab_embeddings
        exp_name=mode
    elif en_vocab_size==10:
        entity_vocab_path=common.path_entity_vocab_10
        entity_embedding_path=common.path_entity_vocab_embeddings_10
        exp_name=mode+"-10"
    elif en_vocab_size==20:
        entity_vocab_path=common.path_entity_vocab_20
        entity_embedding_path=common.path_entity_vocab_embeddings_20
        exp_name=mode+"-20"
    
    data_path=common.path_chunked_dev+"/*"
    
    path_ckpt = common.path_log_root + "/" + exp_name + "/train/checkpoint"
        
    os.system('''
                export CUDA_VISIBLE_DEVICES=%d
                
                python -m run_summarization \
                      --mode=eval \
                      --data_path="%s" \
                      --exp_name=%s \
                      --pointer_gen=%s \
                      --beam_size=%d \
                      --coverage=%s \
                      --use_entity_embedding=%s \
                      --max_enc_steps=%d \
                      --entity_vocab_path="%s" \
                      --entity_embedding_path="%s" \
                      --corpus=%s 
            ''' % (gpu, data_path, exp_name, pointer_gen, beam_size, coverage, use_entity_embedding, 
                   max_enc_steps, entity_vocab_path, entity_embedding_path, corpus))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kbsumm args')
    parser.add_argument('--action', choices=['train', 'beam-search', 'beam-search-batch', 'eval'], default='train')
    parser.add_argument('--mode', choices=['TEO', 'TM'], default="TEO")
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--en_vocab_size', type=int, default=4)
    parser.add_argument('--corpus', choices=['dailymail', 'cnn', 'example'], default='dailymail')
    parser.add_argument('--dec_method', choices=['greedy', 'beam-search'], default='beam-search')
    parser.add_argument('--use_entity_embedding', choices=['True', 'False'], default='True')
    
    
    args = parser.parse_args()
    if args.action == 'train':
        #while True:
            train(mode=args.mode, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus, en_vocab_size=args.en_vocab_size, use_entity_embedding=args.use_entity_embedding)
    elif args.action == 'beam-search':
        beam_search(mode=args.mode, beam_size=args.beam_size, gpu=args.gpu, corpus=args.corpus, is_batch=False, en_vocab_size=args.en_vocab_size, dec_method=args.dec_method, use_entity_embedding=args.use_entity_embedding)
    elif args.action == 'beam-search-batch':
        beam_search(mode=args.mode, beam_size=args.beam_size, gpu=args.gpu, corpus=args.corpus, is_batch=True, en_vocab_size=args.en_vocab_size, dec_method=args.dec_method, use_entity_embedding=args.use_entity_embedding)
    elif args.action == 'eval':
        eval(mode=args.mode, beam_size=args.beam_size, gpu=args.gpu, corpus=args.corpus, en_vocab_size=args.en_vocab_size)





