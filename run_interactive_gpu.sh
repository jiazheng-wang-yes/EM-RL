srun --partition=general \
        --time=4:00:00 \
        --gres=gpu:2   \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=64 \
        --mem=256g \
        --pty /bin/bash