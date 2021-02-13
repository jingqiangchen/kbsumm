import common
import os, re
import argparse

def train(mode="W.O.RL", gcn_level=2, rl_lambda=0.3, batch_size=10, gpu=1, corpus="dailymail", en_vocab_size=4, 
          use_entity_embedding="10", lambda_ee_train=0.0, use_gcn_entity="True"):
    
    if gpu is None:
        gpu=1
    
    if mode=="W.O.RL":
        pass
    
    if en_vocab_size==4:
        entity_vocab_path=common.path_entity_vocab
        entity_embedding_path=common.path_entity_vocab_embeddings
        exp_name = "%s-4-gcn%d" % (mode, gcn_level)
    elif en_vocab_size==10:
        entity_vocab_path=common.path_entity_vocab_10
        entity_embedding_path=common.path_entity_vocab_embeddings_10
        exp_name = "%s-10-gcn%d" % (mode, gcn_level)
    elif en_vocab_size==20:
        entity_vocab_path=common.path_entity_vocab_20
        entity_embedding_path=common.path_entity_vocab_embeddings_20
        exp_name = "%s-20-gcn%d" % (mode, gcn_level)
    
    exp_name = "%s-%.2f" % (exp_name, rl_lambda)
    if use_entity_embedding[0]=='0':
        exp_name += "-noen"
        ext_use_entity_embedding=False
    else:
        ext_use_entity_embedding=True
        
    if lambda_ee_train>0:
        exp_name += "-ee%.1f" % lambda_ee_train
    
    if use_gcn_entity=="False":
        exp_name += "-noge"
    
    entity_vocab_size=en_vocab_size*10000 + 4
    data_path=common.path_chunked+"/train/*"
    if rl_lambda == 0:
        lr=0.01
    else:
        lr=0.0001
    #data_path="/home/test/kbsumm/data/dailymail/chunked/example.bin"
    
    gen_exp_name="TEO-%d" % en_vocab_size
    #gen_exp_name="TEO-4" #% en_vocab_size
    if use_entity_embedding[1]=="0":
        gen_exp_name="TEO-4-noen"
        gen_use_entity_embedding=False
    else:
        gen_use_entity_embedding=True
 
    os.system('''
        export CUDA_VISIBLE_DEVICES=%d
            
        python -m run_gcn \
              --data_path="%s" \
              --exp_name=%s \
              --gcn_level=%d \
              --rl_lambda=%f \
              --batch_size=%d \
              --use_entity_embedding=%s \
              --gen_use_entity_embedding=%s \
              --dec_method=greedy \
              --ckpt_save_model_secs=%d \
              --lr=%f \
              --entity_vocab_path="%s" \
              --entity_embedding_path="%s" \
              --gen_exp_name=%s \
              --entity_vocab_size=%d \
              --lambda_ee_train=%f \
              --use_gcn_entity=%s \
              --corpus=%s 
    ''' % (gpu, data_path, exp_name, gcn_level, rl_lambda, batch_size, ext_use_entity_embedding, gen_use_entity_embedding,
           8000 if rl_lambda==0 else 10000, lr, entity_vocab_path, entity_embedding_path, gen_exp_name, entity_vocab_size, 
           lambda_ee_train, use_gcn_entity, corpus))


def ext_greedy(ext_mode="W.O.RL", gcn_level=2, rl_lambda=0.3, batch_size=10, gpu=1, corpus="dailymail", is_batch=False, en_vocab_size=4, use_entity_embedding="True"):
    
    if gpu is None:
        gpu=1
    
    use_entity_embedding="True"
    
    if ext_mode=="W.O.RL":
        pass
    
    exp_name = "%s-gcn%d" % (ext_mode, gcn_level)
    if rl_lambda>0:
        exp_name = "%s-%.2f" % (exp_name, rl_lambda)
        
    data_path=common.path_chunked+"/dev/*"
    path_ckpt = common.path_log_root_gcn + "/" + exp_name + "/train/checkpoint"
    #data_path="/home/test/kbsumm/data/dailymail/chunked/example.bin"
    
    if en_vocab_size==4:
        entity_vocab_path=common.path_entity_vocab
        entity_embedding_path=common.path_entity_vocab_embeddings
        exp_name = "%s-4-gcn%d" % (ext_mode, gcn_level)
    elif en_vocab_size==10:
        entity_vocab_path=common.path_entity_vocab_10
        entity_embedding_path=common.path_entity_vocab_embeddings_10
        exp_name = "%s-10-gcn%d" % (ext_mode, gcn_level)
    elif en_vocab_size==20:
        entity_vocab_path=common.path_entity_vocab_20
        entity_embedding_path=common.path_entity_vocab_embeddings_20
        exp_name = "%s-20-gcn%d" % (ext_mode, gcn_level)
    
    exp_name = "%s-%.2f" % (exp_name, rl_lambda)
    if use_entity_embedding[0]=='0':
        exp_name += "-noen"
        ext_use_entity_embedding=False
    else:
        ext_use_entity_embedding=True
        
    entity_vocab_size=en_vocab_size*10000 + 4
    
    gen_exp_name="TEO-%d" % en_vocab_size
    #gen_exp_name="" #% en_vocab_size
    if use_entity_embedding[1]=="0":
        gen_exp_name+="TEO-4-noen"
        gen_use_entity_embedding=False
    else:
        gen_use_entity_embedding=True
    
    min_ckpt=100000
    max_ckpt=440000
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
                
            python -m run_gcn \
                  --data_path="%s" \
                  --mode=ext_greedy \
                  --exp_name=%s \
                  --gcn_level=%d \
                  --rl_lambda=%f \
                  --batch_size=%d \
                  --use_entity_embedding=%s \
                  --gen_use_entity_embedding=%s \
                  --single_pass=True \
                  --entity_vocab_path="%s" \
                  --entity_embedding_path="%s" \
                  --gen_exp_name=%s \
                  --entity_vocab_size=%d \
                  --corpus=%s 
        ''' % (gpu, data_path, exp_name, gcn_level, rl_lambda, batch_size, ext_use_entity_embedding, gen_use_entity_embedding, 
               entity_vocab_path, entity_embedding_path, gen_exp_name, entity_vocab_size, corpus))
        
        if not is_batch:
            break


def ext_gen_decode(ext_mode="W.O.RL", gcn_level=2, rl_lambda=0.3, batch_size=10, gpu=1, corpus="dailymail", is_batch=False, write_mode="decode", 
                   en_vocab_size=4, decode_gen=True, use_entity_embedding="True", lambda_ee_train=0.0, use_gcn_entity="True"):
    
    if gpu is None:
        gpu=1
    
    if ext_mode=="W.O.RL":
        pass
    
    #ext_exp_name = "%s-gcn%d" % (ext_mode, gcn_level)
    #ext_exp_name = "%s-%.1f" % (ext_exp_name, rl_lambda)
        
    
    #data_path="/home/test/kbsumm/data/dailymail/chunked/example.bin"
    
    if en_vocab_size==4:
        entity_vocab_path=common.path_entity_vocab
        entity_embedding_path=common.path_entity_vocab_embeddings
        exp_name = "%s-4-gcn%d" % (ext_mode, gcn_level)
    elif en_vocab_size==10:
        entity_vocab_path=common.path_entity_vocab_10
        entity_embedding_path=common.path_entity_vocab_embeddings_10
        exp_name = "%s-10-gcn%d" % (ext_mode, gcn_level)
    elif en_vocab_size==20:
        entity_vocab_path=common.path_entity_vocab_20
        entity_embedding_path=common.path_entity_vocab_embeddings_20
        exp_name = "%s-20-gcn%d" % (ext_mode, gcn_level)
    
    exp_name = "%s-%.2f" % (exp_name, rl_lambda)
    if use_entity_embedding[0]=='0':
        exp_name += "-noen"
        ext_use_entity_embedding=False
    else:
        ext_use_entity_embedding=True
    
    if lambda_ee_train>0:
        exp_name += "-ee%.1f" % lambda_ee_train    
    
    if use_gcn_entity=="False":
        exp_name += "-noge"
        
    entity_vocab_size=en_vocab_size*10000 + 4
    
    gen_exp_name="TEO-%d" % en_vocab_size
    #gen_exp_name="TEO-4" #% en_vocab_size
    if use_entity_embedding[1]=="0":
        gen_exp_name="TEO-4-noen"
        gen_use_entity_embedding=False
    else:
        gen_use_entity_embedding=True
    
    data_path=common.path_chunked+"/dev/*"
    path_ckpt = common.path_log_root_gcn + "/" + exp_name + "/train/checkpoint"
    
    min_ckpt=100000
    max_ckpt=1000000
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
                
            python -m run_gcn \
                  --data_path="%s" \
                  --mode=decode \
                  --exp_name=%s \
                  --gcn_level=%d \
                  --rl_lambda=%f \
                  --batch_size=%d \
                  --use_entity_embedding=%s \
                  --gen_use_entity_embedding=%s \
                  --single_pass=True \
                  --write_mode=%s \
                  --entity_vocab_path="%s" \
                  --entity_embedding_path="%s" \
                  --gen_exp_name=%s \
                  --entity_vocab_size=%d \
                  --decode_gen=%s \
                  --lambda_ee_train=%f \
                  --max_dec_steps=100 \
                  --use_gcn_entity=%s \
                  --corpus=%s 
        ''' % (gpu, data_path, exp_name, gcn_level, rl_lambda, batch_size, ext_use_entity_embedding, gen_use_entity_embedding,
                write_mode, entity_vocab_path, entity_embedding_path, gen_exp_name, entity_vocab_size, decode_gen, lambda_ee_train, use_gcn_entity, corpus))
        
        if not is_batch:
            break

    
def test(mode="W.O.RL", batch_size=10, gpu=1, corpus="dailymail"):
    
    if gpu is None:
        gpu=1
    
    if corpus=="dailymail":
      common.path_chunked=common.path_corpus + "/chunked-gcn"
    elif corpus=="cnn":
      common.path_chunked=common.path_corpus + "/chunked-gcn-cnn"
    
    use_mention_occurs="True"
    use_entity_embedding="True"
    
    if mode=="W.O.RL":
        pass
    
    data_path=common.path_chunked+"/train/*"
    #data_path="/home/test/kbsumm/data/dailymail/chunked/example.bin"
 
    os.system('''
        export CUDA_VISIBLE_DEVICES=%d
            
        python -m run_gcn \
              --data_path="%s" \
              --exp_name=%s \
              --batch_size=%d \
              --use_mention_occurs=%s \
              --use_entity_embedding=%s \
              --mode="test" \
              --corpus=%s 
    ''' % (gpu, data_path, mode, batch_size, use_mention_occurs, use_entity_embedding, corpus))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kbsumm args')
    parser.add_argument('--action', choices=['train', 'beam-search', 'beam-search-batch', 
                                             'ext-greedy', 'ext-greedy-batch', "decode", "decode-batch", 
                                             'test'], default='train')
    parser.add_argument('--mode', choices=['W.O.RL'], default="W.O.RL")
    parser.add_argument('--gcn_level', type=int, default=2)
    parser.add_argument('--rl_lambda', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--en_vocab_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--corpus', choices=['dailymail', 'cnn', 'example'], default='dailymail')
    parser.add_argument('--write_mode', choices=['decode', 'train'], default='decode')
    parser.add_argument('--decode_gen', choices=['True', 'False'], default='True')
    parser.add_argument('--use_entity_embedding', choices=['00', '01', '10', '11'], default='10')
    parser.add_argument('--lambda_ee_train', type=float, default=0)
    parser.add_argument('--use_gcn_entity', choices=['True', 'False'], default='True')
    
    args = parser.parse_args()
    if args.action == 'train':
        train(mode=args.mode, gcn_level=args.gcn_level, rl_lambda=args.rl_lambda, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus, 
              en_vocab_size=args.en_vocab_size, use_entity_embedding=args.use_entity_embedding, lambda_ee_train=args.lambda_ee_train, use_gcn_entity=args.use_gcn_entity)
    elif args.action == 'ext-greedy':
        ext_greedy(mode=args.mode, gcn_level=args.gcn_level, rl_lambda=args.rl_lambda, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus)
    elif args.action == 'ext-greedy-batch':
        ext_greedy(mode=args.mode, gcn_level=args.gcn_level, rl_lambda=args.rl_lambda, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus, is_batch=True)
    elif args.action == 'decode':
        ext_gen_decode(ext_mode=args.mode, gcn_level=args.gcn_level, rl_lambda=args.rl_lambda, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus, 
                       write_mode=args.write_mode, en_vocab_size=args.en_vocab_size, decode_gen=args.decode_gen, use_entity_embedding=args.use_entity_embedding, 
                       lambda_ee_train=args.lambda_ee_train, use_gcn_entity=args.use_gcn_entity)
    elif args.action == 'decode-batch':
        ext_gen_decode(ext_mode=args.mode, gcn_level=args.gcn_level, rl_lambda=args.rl_lambda, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus, 
                       is_batch=True, write_mode=args.write_mode, en_vocab_size=args.en_vocab_size, decode_gen=args.decode_gen, use_entity_embedding=args.use_entity_embedding, 
                       lambda_ee_train=args.lambda_ee_train, use_gcn_entity=args.use_gcn_entity)
        #ext_gen_decode(ext_mode=args.mode, gcn_level=args.gcn_level, rl_lambda=args.rl_lambda, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus, 
        #               is_batch=True, write_mode=args.write_mode, en_vocab_size=args.en_vocab_size, decode_gen=args.decode_gen)
    elif args.action == "test":
        test(mode=args.mode, gcn_level=args.gcn_level, batch_size=args.batch_size, gpu=args.gpu, corpus=args.corpus)





