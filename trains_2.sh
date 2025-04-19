
# /bin/bash
# DONE
#
#
# DONE
# python3 maml.py --omniglot --epoch 7 # 16*6 160 mins
# python3 maml.py --omniglot --k_shot 5 --epoch 7 # 17*10 170 mins
# python3 maml.py --conv --first_order --epoch 15 # 6*15 90 mins
# python3 maml.py --conv --first_order --k_shot 5 --epoch 15 # 8*15 120 mins
# python3 maml.py --conv --epoch 15 # 8*15 120 mins
# python3 maml.py --conv --k_shot 5 --epoch 15 # 21*15 315 mins
# #
python3 maml.py --omniglot --conv --epoch 7 # 40 min * 10 400 mins
python3 maml.py --omniglot --k_shot 5 --conv --epoch 7 # 40 min * 10 400 mins
python3 maml.py --omniglot --n_way 20 --conv --epoch 7 # 46 min * 5 230 mins
python3 maml.py --omniglot --n_way 20 --k_shot 5 --conv --epoch 7 # 89 min * 5 445 mins


# total 1285 mins = 21.4 hours
