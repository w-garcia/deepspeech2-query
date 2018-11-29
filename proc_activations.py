import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

pairs = pickle.load(open('pairs_debug.pkl', 'rb'))

keys = pairs[0]['act_atk'].keys()
res_dict = {}
# err_l2_dict = defaultdict(list)
# og_l2 = defaultdict(list)
change_l2_dict = defaultdict(list)

for p in pairs:
    act1 = p["act_atk"]
    act2 = p["act_og"]

    for k in keys:
        v1 = np.asarray(act1[k])
        v2 = np.asarray(act2[k])
        og_l2 = np.linalg.norm(v2)
        atk_l2 = np.linalg.norm(v1)
        # og_l2[k].append(og_l2)

        err_l2 = np.linalg.norm(v1 - v2)
        # change_l2_dict[k].append(err_l2 / np.average([atk_l2, og_l2]))
        change_l2_dict[k].append(err_l2 / og_l2)

for k in change_l2_dict:
    # v = np.average(err_l2_dict[k]) / np.average(og_l2[k])
    v = np.average(change_l2_dict[k])
    std = np.std(change_l2_dict[k])

    res_dict[k] = (v, std)


vals = [res_dict[k][0] for k in res_dict.keys()]
std = [res_dict[k][1] for k in res_dict.keys()]
y = np.arange(len(change_l2_dict.keys()))
plt.bar(y, vals, align='center', yerr=std)
# plt.xticks(y, ["Convolutional", "Recurrent", "Fully Connected", "Softmax"])
ticks = []
conv_counter = 0
rnn_counter = 0
for k in keys:
    if "conv" in k:
        ticks.append("Conv.{}".format(conv_counter))
        conv_counter += 1
    elif "rnn" in k:
        ticks.append("RNN.{}".format(rnn_counter))
        rnn_counter += 1
    elif "fc" in k:
        ticks.append("FC")
    elif "sm" in k:
        ticks.append("Softmax")

plt.xticks(y, ticks)
plt.ylabel('Relative Change (L2 Norm)')
plt.xlabel('Layer Group')
plt.ylim(0.0, 1.0)
plt.title('L2 Norm per layer (Activation(x) - Activation(x*))')
plt.tight_layout()
plt.show()
