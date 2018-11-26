import librosa

from client import DS2Oracle

fpath = "/home/wgar/deepspeech2/deepspeech2_query/deepspeech2/data/debug/hello.wav"

ds2o = DS2Oracle()
while True:
    x, sr = librosa.load(fpath, sr=16000)
    print(ds2o.transcribe(x, sr))
