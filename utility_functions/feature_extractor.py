"""
SSL feature extractor function for various models
Author: Amit Meghanani
Contact: ameghanani1@sheffield.ac.uk

"""
import wave

import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

## freq HuBERT 16000/320 = 50
## 50 Hz is 20 ms --> 20 ms * 16KHz = 320
## For MFCCs: sr/hop_length 16000/320 = 50


def load_model(model_name,device):
    
    # Select the model:
    if model_name=="HUBERT_BASE":
        bundle = torchaudio.pipelines.HUBERT_BASE
    elif model_name=="WAV2VEC2_BASE":
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
    elif model_name == "WAVLM_BASE":
        bundle = torchaudio.pipelines.WAVLM_BASE
    elif model_name == "WAV2VEC2_XLSR_300M":
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    elif model_name == "HUBERT_LARGE":
        bundle = torchaudio.pipelines.HUBERT_LARGE
    elif model_name == "WAV2VEC2_LARGE":
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE
    elif model_name == "WAVLM_LARGE":
        bundle = torchaudio.pipelines.WAVLM_LARGE
    
    print("loaded model", model_name)


    # Build the model and load pretrained weight.
    model = bundle.get_model().to(device)
    return model,bundle.sample_rate

def SSL_features(path, model, model_sr,layer,device):
    # Load the waveform
    waveform, sample_rate = torchaudio.load(path)

    # Resample audio to the expected sampling rate
    waveform = torchaudio.functional.resample(waveform, sample_rate, model_sr).to(device)

    # Extract acoustic features
    features, _ = model.extract_features(waveform)
    if layer=="all":
        return features
    else:
        # print('lenght of features',len(features))
        # print('layer',layer)
        # print('feature shape',features[layer-1].shape)
        return features[layer-1]

def clip_features(feat,st,ed,layer):
    st_v = int(np.floor(50*st))
    ed_v = int(np.ceil(50*ed))
    if layer=="all":
        for i in range(len(feat)):
            feat[i] = feat[i][:,st_v:st_v + ed_v,:]
    else:
        feat = feat[:,st_v:st_v + ed_v,:]
    return feat

def SSL_features_from_wav(waveform, sample_rate, model, model_sr,layer,device):

    # Resample audio to the expected sampling rate
    waveform = torchaudio.functional.resample(waveform, sample_rate, model_sr).to(device)

    # Extract acoustic features
    features, _ = model.extract_features(waveform)
    if layer=="all":
        return features
    else:
        return features[layer-1]