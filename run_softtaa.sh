DATASET_ARRAY=(sst2 sst5 cola subj trec mr cr proscons)
MODEL_ARRAY=(bert debertav3)
DATA_SIZE_ARRAY=(100 500)
AUG_TYPE=soft_text_autoaugment_searched
BS=32
LR=5e-5
EP=10
LEN=128
DEVICE=cuda:0

clear

for DATASET in ${DATASET_ARRAY[@]}
do

for MODEL in ${MODEL_ARRAY[@]}
do
python main.py --task=classification --job=preprocessing \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN}

for DATA_SIZE in ${DATA_SIZE_ARRAY[@]}
do

CUDA_VISIBLE_DEVICES=0 python main.py --task=classification --job=search \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}
python main.py --task=classification --job=testing \
               --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=${LEN} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}

done # DATA_SIZE

done # MODEL

done # DATASET