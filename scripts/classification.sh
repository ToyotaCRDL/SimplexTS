python -m src.classification.train \
        --dataset_name FMNIST \
        --model_dir models/fmnist &
wait

python -m src.classification.eval \
        --dataset_name FMNIST \
        --model_dir models/fmnist &
wait