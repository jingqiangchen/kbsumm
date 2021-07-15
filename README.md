The code is for the submitted paper titled "An entity-guided text summarization framework with relational heterogeneous graph neural network".  
Some code are borrowed from PG. Thanks for their work.
# Dependancy
JDK for run the entity linking code  
Python2.7 for running the model  
Python3.5 for preprocessing the data  
rdf2vec  
pyrouge 0.1.3
# Dataset 
For CNN/DailyMail, please refer to the link https://github.com/deepmind/rc-data/.  
However, due to the license, NYT(The New York Times Annotated Corpus) can only be available from LDC. And we follow the preprocessing code of Durrett et al. (2016) to get the NYT50 datasets.
# Preprocess
For CNN/DM, please use the script preprocess-python35/preprocess.py.  
For NYT50, please use the script preprocess-python35/preprocess-nyt50.py.  
To link entities in text to YAGO2, please use the java code entity-linking-java/KBSumm.java
# Run experiments
To train the generator, please run the command: "python main_generator.py --batch_size 15 --use_entity_embedding False" .  
To train the extractor, please run the command: "python main_extractor.py --batch_size 10 --use_entity_embedding 10 --rl_lambda 0.0 --gcn_level 2 --lambda_ee_train 0.1"
