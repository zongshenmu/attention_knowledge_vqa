## Script for downloading data

# GloVe Vectors
wget -P rawdata http://nlp.stanford.edu/data/glove.6B.zip
unzip rawdata/glove.6B.zip -d rawdata/glove
rm rawdata/glove.6B.zip
mv rawdata/glove/glove.6B.300d.txt rawdata/
rm -r rawdata/glove/

# Questions
wget -P rawdata http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip rawdata/v2_Questions_Train_mscoco.zip -d rawdata
rm rawdata/v2_Questions_Train_mscoco.zip

wget -P rawdata http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip rawdata/v2_Questions_Val_mscoco.zip -d rawdata
rm rawdata/v2_Questions_Val_mscoco.zip

wget -P rawdata http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip rawdata/v2_Questions_Test_mscoco.zip -d rawdata
rm rawdata/v2_Questions_Test_mscoco.zip

# Annotations
wget -P rawdata http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip rawdata/v2_Annotations_Train_mscoco.zip -d rawdata
rm rawdata/v2_Annotations_Train_mscoco.zip

wget -P rawdata http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip rawdata/v2_Annotations_Val_mscoco.zip -d rawdata
rm rawdata/v2_Annotations_Val_mscoco.zip

# Image Features
wget -P rawdata https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip rawdata/trainval_36.zip -d rawdata
rm rawdata/trainval_36.zip
mv trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv rawdata/
rm -r trainval_36
