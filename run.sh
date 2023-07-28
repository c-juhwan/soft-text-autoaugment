DATASET=sst2
BS=32
LR=5e-5
EP=5

clear

MODEL=bert
DATA_SIZE=full
# python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --data_subsample_size=${DATA_SIZE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --data_subsample_size=${DATA_SIZE}

DATA_SIZE=1000
AUG_TYPE=aeda
# python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}

AUG_TYPE=soft_text_autoaugment_searched
python main.py --task=classification --job=search --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --data_subsample_size=${DATA_SIZE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE} --data_subsample_size=${DATA_SIZE}