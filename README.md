# SSL_AWE, coming soon!
EACL 2024 -  **[Improving Acoustic Word Embeddings through Correspondence Training of Self-supervised Speech Representations](https://aclanthology.org/2024.eacl-long.118.pdf)**

## Download data
Download force-aligned dataset (timestamps, word list): **[MLS_force_aligned](https://drive.google.com/file/d/13bVpExtoQwxplFiQVvUALDvDVWjNsdHb/view?usp=sharing)** \
Download corresponding speech corpora: https://www.openslr.org/94/ \
Note: for english speech corpora, please download the partaa only (due to huge amount of data): https://dl.fbaipublicfiles.com/mls/mls_english_parts_list.txt

#### Note
If you want to force align the dataset yourself, you may use  the following commands to do so via [MFA toolkit](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) \
Please arrange the datafiles in the required format used in mfa directory structure. You may use the code in ``` python prepare_data.py``` with some modification to do that.
```
conda activaet mfa ## create an environment with MFA toolkit installed
mfa models download acoustic english_mfa
mfa models download dictionary english_us_mfa

mfa align --clean ....your path/MLS_processed/mls_english/train/ english_us_mfa english_mfa ....your path/MLS_force_aligned/mls_english/train/ --output_format=csv --beam 100 --retry_beam 400
mfa align --clean ....your path/MLS_processed/mls_english/dev/ english_us_mfa english_mfa ....your path/MLS_force_aligned/mls_english/dev/ --output_format=csv --beam 100 --retry_beam 400
mfa align --clean ....your path/MLS_processed/mls_english/test/ english_us_mfa english_mfa ....your path/MLS_force_aligned/mls_english/test/ --output_format=csv --beam 100 --retry_beam 400
```

## Prepara metadata for training
Already prepared metadata is available at in **/metadata** folder OR \
Use the code ```python prepare_metadata.py``` to get train_metadata.csv, dev_metadata.csv, and test_metadata.csv for all the langauges separately. Change the paths in the code for various languages.

## Extract and store SSL features
### With context
For HuBERT: ```python extract_ssl.py @config_files/extract_hubert.txt```, For Wav2vec: ```python extract_ssl.py @config_files/extract_wav2vec2.txt```, For WavLM: ```python extract_ssl.py @config_files/extract_wavlm.txt```
### Without context
For HuBERT: ```python extract_ssl_woc.py @config_files/extract_hubert.txt```, For Wav2vec: ```python extract_ssl_woc.py @config_files/extract_wav2vec2.txt```, For WavLM: ```python extract_ssl_woc.py @config_files/extract_wavlm.txt```

### For MFCC
```python extract_mfcc.py @config_files/extract_mfcc.txt```

## Run AE and CAE models for various input features

For HuBERT: ```python cae.py @config_files/cae_hubert.txt```, For wav2vec2: ```python cae.py @config_files/cae_wav2vec2.txt```, For WavLM: ```python cae.py @config_files/cae_wavlm.txt```, for MFCC: ```python cae.py @config_files/cae_mfcc.txt```

Similarly for AE models.

#### Note:
Change the ```--metadata_file``` path with _woc (without context features) and _wc (with context) in /config_files/cae_** or /config_files/ae_**

## Evaluate the models
```python eval_awe.py @config_files/eval_awe.txt```
#### Note:
Change the ```--model_weights``` and ```--metadata_file``` according to Langauge and Model you want to evalaute for word-discrimination task.

#### Evaluate with pooling mechanism:
```python pooling_eval.py```, please change the ```metadata_filepath``` inside the code as per your need.
