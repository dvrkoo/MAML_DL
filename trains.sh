# Target: ~30,000 Steps for all experiments.

# ============================
# Omniglot 5-way FCNet (Non-Conv)
# Paper HParams: inner_lr=0.4, update_step=1, batch_size=32, update_step_test=3
# Epochs = ceil( (30000 * 32) / 60000 ) = 16
# ============================

# Command 1: Omni, 5-way, 1-shot, FCNet
# Steps ≈ 16 * (60000 / 32) = 30000
python3 maml.py --omniglot \
    --inner_lr 0.4 --update_step 1 --batch_size 32 --update_step_test 3 \
    --epoch 16 # Target Steps: ~30000

# Command 2: Omni, 5-way, 5-shot, FCNet
# Steps ≈ 16 * (60000 / 32) = 30000
python3 maml.py --omniglot --k_shot 5 \
    --inner_lr 0.4 --update_step 1 --batch_size 32 --update_step_test 3 \
    --epoch 16 # Target Steps: ~30000

# ============================
# MiniImageNet ConvNet (Assumed 5-way)
# Paper HParams: inner_lr=0.01, update_step=5, update_step_test=10
# Paper BS: 4 (1-shot), 2 (5-shot).
# ============================

# Command 3: Mini, 5-way, 1-shot, ConvNet, First Order
# Epochs = ceil( (30000 * 4) / 10000 ) = 12
# Steps ≈ 12 * (10000 / 4) = 30000
python3 maml.py --conv --first_order \
    --inner_lr 0.01 --update_step 5 --batch_size 4 --update_step_test 10 \
    --epoch 12 # Target Steps: ~30000

# Command 4: Mini, 5-way, 5-shot, ConvNet, First Order
# Epochs = ceil( (30000 * 2) / 10000 ) = 6
# Steps ≈ 6 * (10000 / 2) = 30000
python3 maml.py --conv --first_order --k_shot 5 \
    --inner_lr 0.01 --update_step 5 --batch_size 2 --update_step_test 10 \
    --epoch 6 # Target Steps: ~30000

# # Command 5: Mini, 5-way, 1-shot, ConvNet, Second Order
# # Epochs = ceil( (30000 * 4) / 10000 ) = 12
# # Steps ≈ 12 * (10000 / 4) = 30000
python3 maml.py --conv \
    --inner_lr 0.01 --update_step 5 --batch_size 4 --update_step_test 10 \
    --epoch 12 # Target Steps: ~30000

# # Command 6: Mini, 5-way, 5-shot, ConvNet, Second Order
# # Epochs = ceil( (30000 * 2) / 10000 ) = 6
# # Steps ≈ 6 * (10000 / 2) = 30000
python3 maml.py --conv --k_shot 5 \
    --inner_lr 0.01 --update_step 5 --batch_size 2 --update_step_test 10 \
    --epoch 6 # Target Steps: ~30000

# # ============================
# # Omniglot 5-way ConvNet
# # Paper HParams: inner_lr=0.4, update_step=1, batch_size=32, update_step_test=3
# # Epochs = ceil( (30000 * 32) / 60000 ) = 16
# # ============================
#
# # Command 7: Omni, 5-way, 1-shot, ConvNet
# # Steps ≈ 16 * (60000 / 32) = 30000
python3 maml.py --omniglot --conv \
    --inner_lr 0.4 --update_step 1 --batch_size 32 --update_step_test 3 \
    --epoch 16 # Target Steps: ~30000

# Command 8: Omni, 5-way, 5-shot, ConvNet
# Steps ≈ 16 * (60000 / 32) = 30000
python3 maml.py --omniglot --k_shot 5 --conv \
    --inner_lr 0.4 --update_step 1 --batch_size 32 --update_step_test 3 \
    --epoch 16 # Target Steps: ~30000
# #
# # ============================
# Omniglot 20-way ConvNet
# Paper HParams: inner_lr=0.1, update_step=5, update_step_test=5
# Using CUSTOM batch_size=8 (Paper uses 16)
 Epochs = ceil( (30000 * 8) / 60000 ) = 4
# ============================
#
# # Command 9: Omni, 20-way, 1-shot, ConvNet
# # Steps ≈ 4 * (60000 / 8) = 30000
python3 maml.py --omniglot --n_way 20 --conv \
    --inner_lr 0.1 --update_step 5 --batch_size 8 --update_step_test 5 \
    --epoch 4 # Target Steps: ~30000 (NOTE: Using BS=8, not paper's 16)
#
# # Command 10: Omni, 20-way, 5-shot, ConvNet
# # Steps ≈ 4 * (60000 / 8) = 30000
python3 maml.py --omniglot --n_way 20 --k_shot 5 --conv \
    --inner_lr 0.1 --update_step 5 --batch_size 8 --update_step_test 5 \
    --epoch 4 # Target Steps: ~30000 (NOTE: Using BS=8, not paper's 16)
