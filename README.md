Model
---
the vqa model framework illustrates as blow
![image](https://raw.githubusercontent.com/zongshenmu/attention_knowledge_vqa/master/framework.png)
  
# first step
```
run: download.sh
```
download rawdata include:
>1)question
>>v2_OpenEnded_mscoco_train2014_questions.json
>>v2_OpenEnded_mscoco_val2014_questions.json
>>v2_OpenEnded_mscoco_test2015_questions.json
>>v2_OpenEnded_mscoco_test-dev2015_questions.json

>2)answer
>>v2_mscoco_train2014_annotations.json
>>v2_mscoco_val2014_annotations.json

>3)image id
>>train_ids.pkl
>>val_ids.pkl

>4)image salinet region features
>>trainval_resnet101_faster_rcnn_genome_36.tsv

>5)image bounding boxes info
>>instances_train2014.json
>>instances_val2014.json

# second step
```
run the codes in the /preprocess: process.sh
```
preprocess the rawdata turning to maturedata

# thrid step
```
run: python3.6 train.py
```
train the bottom-up and top with knowledge base VQA model in the python3.6 and tensorflwo1.3(cpu or gpu) environment

# final step
if you want test or visualise result, you can refer the codes in the /postprecess and /test


# knowledge.json
handcraft knowledge include mscoco dataset 80 object labels information from wikipedia

#knowledge document vector(docid.json, doc_embeddings.pkl)
refer my another repository:doc2vec
