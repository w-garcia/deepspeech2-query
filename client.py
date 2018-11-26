import argparse
import librosa
import Pyro4

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DS2Oracle:
    def __init__(self):
        self.DS2ASR = Pyro4.Proxy("PYRONAME:DS2ASR")

    def transcribe(self, audio_data, sample_rate=16000):
        try:
            if type(audio_data) is list:
                transcription = self.DS2ASR.get_transcription(audio_data, sr=sample_rate)
            else:
                transcription = self.DS2ASR.get_transcription(audio_data.tolist(), sr=sample_rate)
            return transcription
        except Exception as e:
            print(e)
            logger.error("The query system crashed. Restart <r> or quit <q>:")
            inp = input()
            if inp == 'q' or inp == 'Q':
                return ""
            else:
                self.DS2ASR = Pyro4.Proxy("PYRONAME:DS2ASR")
                return self.transcribe(audio_data, sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser.add_argument('--uri',
                        help='URI of server', required=True)
    # /home/wgar/deepspeech2/deepspeech2-query/deepspeech2/data/debug/hello.wav

    args = parser.parse_args()
    uri = args.uri
    while True:
        try:
            DS2ASR = Pyro4.Proxy(uri)
            while True:
                fpath = "/home/wgar/deepspeech2/deepspeech2-query/deepspeech2/data/debug/hello.wav" #input("PATH: ")
                x, sr = librosa.load(fpath, sr=16000)
                print(DS2ASR.get_transcription(x.tolist(), sr=sr))

        except Exception as e:
            print(e)
            uri = input("The query system crashed. Enter new URI to restart, or Q to quit:\n")
            if uri == 'q' or uri == 'Q':
                break
