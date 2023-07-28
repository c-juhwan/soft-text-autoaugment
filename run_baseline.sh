DATASET=sst2
BS=32
LR=5e-5
EP=5

clear

### BERT ###
MODEL=bert
# AUG_TYPE=none
# python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

# AUG_TYPE=hard_eda
# python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

# AUG_TYPE=soft_eda
# python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
# python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
# python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

### DEBERTAV3 ###
MODEL=debertav3
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}


##############
#### TREC
##############
clear

DATASET=trec
BS=32
LR=5e-5
EP=5

clear

### BERT ###
MODEL=bert
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

### DEBERTAV3 ###
MODEL=debertav3
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}


##############
#### COLA
##############
clear

DATASET=cola
BS=32
LR=5e-5
EP=5

clear

### BERT ###
MODEL=bert
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

### DEBERTAV3 ###
MODEL=debertav3
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}


##############
#### subj
##############
clear

DATASET=subj
BS=32
LR=5e-5
EP=5

clear

### BERT ###
MODEL=bert
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

### DEBERTAV3 ###
MODEL=debertav3
AUG_TYPE=none
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=hard_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=soft_eda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}

AUG_TYPE=aeda
python main.py --task=classification --job=augment --task_dataset=${DATASET} --model_type=${MODEL} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --augmentation_type=${AUG_TYPE}


##############
#### TREC
##############
clear