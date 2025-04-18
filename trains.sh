# /bin/bash
# DONE
#
#
# DONE
python3 maml.py --omniglot # 16*5 80 mins
python3 maml.py --omniglot --k_shot 5 # 17*5 85 mins
python3 maml.py --conv --first_order --epoch 10 # 6*5 30 mins
python3 maml.py --conv --first_order --k_shot 5 --epoch 10 # 8*5 40 mins
python3 maml.py --conv --epoch 10 # 8*5 40 mins
python3 maml.py --conv --k_shot 5 --epoch 10 # 21*5 105 mins
#
python3 maml.py --omniglot --conv # 40 min * 5  200mins
python3 maml.py --omniglot --k_shot 5 --conv # 40 min * 5 200mins
python3 maml.py --omniglot --n_way 20 --conv # 46 min * 5 230 mins
python3 maml.py --omniglot --n_way 20 --k_shot 5 --conv # 89 min * 5 445 mins


# total 1285 mins = 21.4 hours
