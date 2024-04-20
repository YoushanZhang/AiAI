# Config
Here are the config files used to train the single/multi-speaker TTS models.
4 different configurations are given:
- LibriTTS: suggested configuration for LibriTTS dataset.


Setup Your model.yaml, train.yaml and preprocess.yml before everything inside config

#Preprocessing

python3 prepare_align.py config/{Your voice Dataset}/preprocess.yaml

to align the corpus and then run the preprocessing script - Followed by 

python3 preprocess.py config/{Your voice Dataset}/preprocess.yaml

#Training
python3 train.py -p config/{Your voice Dataset}/preprocess.yaml -m config/{Your voice Dataset}/model.yaml -t config/{Your voice Dataset}/train.yaml


#Inference
For pretrained models put them in output/ckpt/{Your voice Dataset}/
python3 synthesize.py --text "YOUR DESIRED TEXT" --restore_step 900000 --mode single -p config/{Your voice Dataset}/preprocess.yaml -m config/{Your voice Dataset}/model.yaml -t config/{Your voice Dataset}/train.yaml