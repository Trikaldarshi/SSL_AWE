# SSL_AWE, coming soon!
EACL 2024 -  **[Improving Acoustic Word Embeddings through Correspondence Training of Self-supervised Speech Representations](https://aclanthology.org/2024.eacl-long.118.pdf)**

Download force-aligned dataset (timestamps, word list): **[MLS_force_aligned](https://drive.google.com/file/d/13bVpExtoQwxplFiQVvUALDvDVWjNsdHb/view?usp=sharing)** \
Download corresponding speech corpora: https://www.openslr.org/94/ \
Note: for english speech corpora, please download the partaa only (due to huge amount of data): https://dl.fbaipublicfiles.com/mls/mls_english_parts_list.txt

#### Note
If you want to force align the dataset yourself, you may use  the following commands to do so via [MFA toolkit](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) \
Please arrange the datafiles in the required format used in mfa directory structure. You may use the code in ``` python prepare_data.py``` with some modification.
```
conda activaet mfa ## create an environment with MFA toolkit installed
mfa models download acoustic english_mfa
mfa models download dictionary english_us_mfa

mfa align --clean ....your path/MLS_processed/mls_english/train/ english_us_mfa english_mfa ....your path/MLS_force_aligned/mls_english/train/ --output_format=csv --beam 100 --retry_beam 400
mfa align --clean ....your path/MLS_processed/mls_english/dev/ english_us_mfa english_mfa ....your path/MLS_force_aligned/mls_english/dev/ --output_format=csv --beam 100 --retry_beam 400
mfa align --clean ....your path/MLS_processed/mls_english/test/ english_us_mfa english_mfa ....your path/MLS_force_aligned/mls_english/test/ --output_format=csv --beam 100 --retry_beam 400
```
