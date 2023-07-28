DATASET_ARRAY=(sst2 cola trec subj imdb)
MODEL_ARRAY=(bert debertav3)
DATA_SIZE_ARRAY=(full 100 500 1000 2000)
AUG_ARRAY=(none hard_eda soft_eda aeda)
BS=32
LR=5e-5
EP=10
DEVICE=cuda:3

clear

for DATASET in ${DATASET_ARRAY[@]}
do
if [ ${DATASET} == "imdb" ]
then
    LEN=512
else
    LEN=128
fi

for MODEL in ${MODEL_ARRAY[@]}
do
python main.py --task=classification --job=preprocessing \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN}

for DATA_SIZE in ${DATA_SIZE_ARRAY[@]}
do

for AUG_TYPE in ${AUG_ARRAY[@]}
do
python main.py --task=classification --job=augment \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}
python main.py --task=classification --job=training \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}
python main.py --task=classification --job=testing \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}

done # AUG_TYPE

done # DATA_SIZE

done # MODEL

done # DATASET