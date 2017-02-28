""" Loads the TIMIT data into a directory appropriately. """

import numpy as np
import os
from os.path import join
import python_speech_features
import scipy.io.wavfile as wav
import shutil
import subprocess

import config

def create_raw_data():
    """ Copies the original TIMIT data to a working directory and does basic
    preprocessing such as removing phone timestamps and converting NIST files
    to WAV."""

    def preprocess_phones(org_path, tgt_path):
        """ Takes the org_path .phn file, preprocesses and writes to tgt_path."""
        with open(org_path) as in_f, open(tgt_path, "w") as out_f:
            phones = []
            for line in in_f:
                phones.append(line.strip().split()[-1])
            print(" ".join(phones), file=out_f)

    def sph2wav(sphere_path, wav_path):
        """ Calls sox to convert the sphere file to wav."""
        args = [config.SOX_PATH, sphere_path, wav_path]
        subprocess.run(args)

    utter_id = 0
    for root, dirs, fns in os.walk(config.ORG_DIR):
        for fn in fns:
            org_path = join(root, fn)

            if fn.endswith("phn"):
                tgt_path = join(config.TGT_DIR, "char_y", "%d.phn" % utter_id)
                preprocess_phones(org_path, tgt_path)

                # Address the corresponding WAV file.
                tgt_path = join(config.TGT_DIR, "feats", "%d.wav" % utter_id)
                org_path = org_path[:-3]+"wav"
                sphere_path = tgt_path[:-3]+"sph" # It's actually in sphere format
                shutil.copyfile(org_path, sphere_path)
                sph2wav(sphere_path, tgt_path)

                utter_id += 1

def phn2npy():
    """ Converts the *.phn files to *.npy, which are 1d numpy arrays of integers."""
    pass

def feat_extract():
    """ Extracts features from WAV files and puts them in 2e numpy arrays of floats."""

    def extract_energy(rate, sig):
        """ Extracts the energy of frames. """
        mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
        energy_row_vec = mfcc[:,0]
        energy_col_vec = energy_row_vec[:, np.newaxis]
        return energy_col_vec

    def feature_extraction(wav_path):
        """ Currently grabs log Mel filterbank, deltas and double deltas."""

        def collapse(utterance):
            """ Converts timit utterance into an array of format (freq, time). Except
            where Freq is Freqxnum_deltas, so usually freq*3. Essentially multiple
            channels are collapsed to one"""

            #import pdb; pdb.set_trace()

            swapped = np.swapaxes(utterance,0,1)
            concatenated = np.concatenate(swapped,axis=1)
            collapsed = np.swapaxes(concatenated,0,1)
            return collapsed

        (rate, sig) = wav.read(wav_path)
        fbank_feat = python_speech_features.logfbank(sig, rate, nfilt=40)
        energy = extract_energy(rate, sig)
        feat = np.hstack([energy, fbank_feat])
        delta_feat = python_speech_features.delta(feat, 2)
        delta_delta_feat = python_speech_features.delta(delta_feat, 2)
        l = [feat, delta_feat, delta_delta_feat]
        all_feats = np.array(l)
        # Make time the first dimension for easy length normalization padding later.
        all_feats = np.swapaxes(all_feats, 0, 1)
        all_feats = np.swapaxes(all_feats, 1, 2)

        collapsed_feats = collapse(all_feats)

        # Log Mel Filterbank, with delta, and double delta
        feat_fn = wav_path[:-3] + "npy"
        np.save(feat_fn, collapsed_feats)


    for root, dirs, fns in os.walk(join(config.TGT_DIR, "feats")):
        for fn in fns:
            print("Processing utterance %s" % fn)
            if fn.endswith(".wav"):
                feature_extraction(join(root, fn))

    pass

if __name__ == "__main__":
    #create_raw_data()
    feat_extract()
