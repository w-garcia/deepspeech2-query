import librosa
import os
import fnmatch
import argparse
import pickle

from client import DS2Oracle

opt = argparse.ArgumentParser()
opt.add_argument("--data", required=True)
args = opt.parse_args()

ds2o = DS2Oracle(ns="localhost", hk="7ujm")
atks = []
ogs = []

for root, dirs, filenames in os.walk(args.data):
    for name_ in fnmatch.filter(filenames, "*atk.*"):
        atks.append({'name': name_, 'fpath': os.path.join(root, name_)})
    for name_ in fnmatch.filter(filenames, "*og.*"):
        ogs.append({'name': name_, 'fpath': os.path.join(root, name_)})

atks.sort(key=lambda x: x['name'])
ogs.sort(key=lambda  x: x['name'])

pairs = []

for atk, og in zip(atks, ogs):
    print(atk['name'], " -> ", og['name'])
    x1, sr1 = librosa.load(atk['fpath'], sr=16000)
    act1 = ds2o.get_activations(x1, sr1)

    x2, sr2 = librosa.load(og['fpath'], sr=16000)
    act2 = ds2o.get_activations(x2, sr2)
    pairs.append({
        "fpath_atk": atk['fpath'], "act_atk": act1,
        "fpath_og": og['fpath'], "act_og": act2
    })

pickle.dump(pairs, open('pairs_debug.pkl', 'wb'))
print("Write pairs_debug.pkl")


