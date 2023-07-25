DATASET=sst2
BS=32
LR=5e-5
EP=5

clear

MODEL=bert
#python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
#python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
#python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

python main.py --task=classification --job=search --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}