import Pyro4
import argparse
import warnings
import os.path
import tempfile
import librosa
import numpy as np

warnings.simplefilter('ignore')

from deepspeech2.model import DeepSpeech
from deepspeech2.decoder import GreedyDecoder
from deepspeech2.data.data_loader import SpectrogramParser
from torch.autograd import Variable

"""
Make sure to run:
python -m Pyro4.naming

Prior to using.
"""

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--hostname', required=True, help='Server hostname/IP to use.')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top_paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')
args = parser.parse_args()

print("Loading model...")

model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
model.eval()


@Pyro4.expose
class DeepSpeech2ASR(object):
    def __init__(self):
        self.model = model

        labels = DeepSpeech.get_labels(self.model)
        audio_conf = DeepSpeech.get_audio_conf(self.model)

        if args.decoder == "beam":
            from deepspeech2.decoder import BeamCTCDecoder

            self.decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                          cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                          beam_width=args.beam_width, num_processes=args.lm_workers)
        else:
            self.decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

        self.parser = SpectrogramParser(audio_conf, normalize=True)
        self.queries = 0

    def decode_results(self, decoded_output, decoded_offsets):
        results = {
            "output": [],
            "_meta": {
                "acoustic_model": {
                    "name": os.path.basename(args.model_path),
                    **DeepSpeech.get_meta(self.model)
                },
                "language_model": {
                    "name": os.path.basename(args.lm_path) if args.lm_path else None,
                },
                "decoder": {
                    "lm": args.lm_path is not None,
                    "alpha": args.alpha if args.lm_path is not None else None,
                    "beta": args.beta if args.lm_path is not None else None,
                    "type": args.decoder,
                }
            }
        }

        for b in range(len(decoded_output)):
            for pi in range(min(args.top_paths, len(decoded_output[b]))):
                result = {'transcription': decoded_output[b][pi]}
                if args.offsets:
                    result['offsets'] = decoded_offsets[b][pi]
                results['output'].append(result)
        return results

    def get_transcription(self, audio_data, sr=16000):
        if type(audio_data) is str:
            spect = self.parser.parse_audio(np.array(audio_data)).contiguous()
            spect = spect.view(1, 1, spect.size(0), spect.size(1))
            out = self.model(Variable(spect, volatile=True))
            out = out.transpose(0, 1)  # TxNxH
            decoded_output, decoded_offsets = self.decoder.decode(out.data)
            res = self.decode_results(decoded_output, decoded_offsets)
            self.queries += 1
            transcription = res['output'][0]['transcription']
            print("Processed query#{}, Result: {}".format(self.queries, transcription))
            # print(json.dumps(res))
            return transcription

        else:
            with tempfile.NamedTemporaryFile() as f:
                try:
                    librosa.output.write_wav(f.name, np.array(audio_data), sr)

                    spect = self.parser.parse_audio(f.name).contiguous()
                    spect = spect.view(1, 1, spect.size(0), spect.size(1))
                    out = self.model(Variable(spect, volatile=True))
                    out = out.transpose(0, 1)  # TxNxH
                    decoded_output, decoded_offsets = self.decoder.decode(out.data)
                    res = self.decode_results(decoded_output, decoded_offsets)
                    self.queries += 1
                    transcription = res['output'][0]['transcription']
                    print("Processed query#{}, Result: {}".format(self.queries, transcription))
                    # print(json.dumps(res))
                    return transcription
                except Exception as e:
                    print(e)
                    return ""


def main():
    hk = input("HMAC key: ")
    ns = Pyro4.locateNS(hmac_key=hk)
    daemon = Pyro4.Daemon(host=args.hostname)
    daemon._pyroHmacKey = hk
    uri = daemon.register(DeepSpeech2ASR)
    ns.register("DS2ASR", uri)
    print("Daemon ready.")
    # daemon = Pyro4.Daemon.serveSimple(
    #     {
    #         DeepSpeech2ASR: "DS2ASR"
    #     },
    #     ns=True,
    #     host="10.136.17.175"
    # )
    daemon.requestLoop()


if __name__ == '__main__':
    main()
