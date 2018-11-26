import argparse
import librosa
import Pyro4

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ORACLENAME = "PYRONAME:DS2ASR"


class DS2Oracle:
    """
    Expects numpy or list-type (not string)
    """
    def __init__(self):
        self.DS2ASR = Pyro4.Proxy(ORACLENAME)

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
                self.DS2ASR = Pyro4.Proxy(ORACLENAME)
                return self.transcribe(audio_data, sample_rate)
