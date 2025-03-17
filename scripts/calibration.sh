for beta in 0.2 0.3 0.4 0.5
do
    python -m src.calibration.train \
        --dataset_name FMNIST \
        --beta ${beta} \
        --model_dir models/fmnist \
        --results_dir results/fmnist &
done
wait

for beta in 0.6 0.7 0.8 0.9
do
    python -m src.calibration.train \
        --dataset_name FMNIST \
        --beta ${beta} \
        --model_dir models/fmnist \
        --results_dir results/fmnist &
done
wait

for beta in 1.0 1.1 1.2 1.3
do
    python -m src.calibration.train \
        --dataset_name FMNIST \
        --beta ${beta} \
        --model_dir models/fmnist \
        --results_dir results/fmnist &
done
wait

for beta in 1.4 1.5 1.6 1.7
do
    python -m src.calibration.train \
        --dataset_name FMNIST \
        --beta ${beta} \
        --model_dir models/fmnist \
        --results_dir results/fmnist &
done
wait

for beta in 1.8 1.9 2.0
do
    python -m src.calibration.train \
        --dataset_name FMNIST \
        --beta ${beta} \
        --model_dir models/fmnist \
        --results_dir results/fmnist &
done
wait

python -m src.calibration.tuning \
    --results_dir results/fmnist &
wait

python -m src.calibration.eval \
    --dataset_name FMNIST \
    --model_dir models/fmnist \
    --results_dir results/fmnist &
wait