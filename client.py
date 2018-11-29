import argparse
import librosa
import Pyro4

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ORACLENAME = "DS2ASR"


class DS2Oracle:
    """
    Expects numpy or list-type (not string)
    """
    def __init__(self, hk=None, ns=None):
        if ns is None:
            ns = input("Server IP: ")
        if hk is None:
            hk = input("HMAC key: ")

        self.ns = Pyro4.locateNS(ns, hmac_key=hk, broadcast=False)
        uri = self.ns.lookup(ORACLENAME)
        self.DS2ASR = Pyro4.Proxy(uri)
        self.DS2ASR._pyroHmacKey = hk

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
                hk = input("HMAC key: ")
                self.ns = Pyro4.locateNS(input("Server IP: "), hmac_key=hk, broadcast=False)
                uri = self.ns.lookup(ORACLENAME)
                self.DS2ASR = Pyro4.Proxy(uri)
                self.DS2ASR._pyroHmacKey = hk
                return self.transcribe(audio_data, sample_rate)

    def get_activations(self, audio_data, sample_rate=16000):
        try:
            activation_dict = self.DS2ASR.get_activation_dict(audio_data.tolist(), sr=sample_rate)
            return activation_dict
        except Exception as e:
            print(e)
            logger.error("The query system crashed. Restart <r> or quit <q>:")
            inp = input()
            if inp == 'q' or inp == 'Q':
                return ""
            else:
                hk = input("HMAC key: ")
                self.ns = Pyro4.locateNS(input("Server IP: "), hmac_key=hk, broadcast=False)
                uri = self.ns.lookup(ORACLENAME)
                self.DS2ASR = Pyro4.Proxy(uri)
                self.DS2ASR._pyroHmacKey = hk
                return self.get_activations(audio_data, sample_rate)
