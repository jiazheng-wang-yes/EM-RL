srun --partition=general \
        --time=1:00:00 \
        --gres=gpu:h100:4   \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=64 \
        --mem=256g \
        --pty /bin/bash