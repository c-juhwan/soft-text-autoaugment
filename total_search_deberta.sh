DATASET_ARRAY=(sst2 sst5 cola trec subj) # (agnews imdb yelp_full)
MODEL_TYPE=debertav3
DATA_SIZE_ARRAY=(100 500 1000)
AUG_TYPE=soft_text_autoaugment_searched
BS=32
LR=5e-5
EP=10
DEVICE=cuda:2

clear

###### SEARCH ######
for DATASET in ${DATASET_ARRAY[@]}
do

if [ ${DATASET} == "imdb" ]; then
    LEN=512
elif [ ${DATASET} == "yelp_full" ]; then
    LEN=512
else
    LEN=128
fi

python main.py --task=classification --job=preprocessing \
               --task_dataset=${DATASET} --model_type=${MODEL_TYPE} --max_seq_len=${LEN}

for DATA_SIZE in ${DATA_SIZE_ARRAY[@]}
do

CUDA_VISIBLE_DEVICES=2 python main.py --task=classification --job=search \
               --task_dataset=${DATASET} --model_type=${MODEL_TYPE} --max_seq_len=${LEN} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}

done # DATA_SIZE

done # DATASET

###### TEST ######
for DATASET in ${DATASET_ARRAY[@]}
do

if [ ${DATASET} == "imdb" ]; then
    LEN=512
elif [ ${DATASET} == "yelp_full" ]; then
    LEN=512
else
    LEN=128
fi

for DATA_SIZE in ${DATA_SIZE_ARRAY[@]}
do

CUDA_VISIBLE_DEVICES=2 python main.py --task=classification --job=testing \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}

done # DATA_SIZE

done # DATASET