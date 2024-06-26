import os
import glob
import random
from sklearn.utils import shuffle


## Example for Polish for dev set, 4 changes are required in the code for different languages and different sets

# check the directory and if does not exist, create it
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

base_dir = ''
data_dir = '....your path/MLS/mls_polish/test/audio/'                    ## 1. CHANGE HERE
target_dir = '....your path/MLS_processed/mls_polish/test/'              ## 2. CHANGE HERE


# max_count = 25000 #  for train            ## 3. CHANGE HERE
max_count = 500 # for dev or test 


check_dir(base_dir + target_dir)

# get the list of wav files at location base_dir + data_dir
flac_paths = glob.glob(base_dir + data_dir + '*/**/*.flac')

# read the transcirpt file at location '....your path/MLS/mls_polish/train/transcripts.txt'
with open(base_dir + '....your path/MLS/mls_polish/test/transcripts.txt','r') as f:         ## 4. CHANGE HERE
    transcripts = f.readlines()

print("total audio files: ", len(flac_paths), "; total transcripts: ", len(transcripts))


flac_paths = shuffle(flac_paths, random_state=42)


# create a dictionary of audio file name and its transcript
transcript_dict = {}
count = 0
for transcript in transcripts:
    transcript = transcript.strip()
    transcript = transcript.split('\t')
    transcript_dict[transcript[0]] = transcript[1:]

count = 0
for i in flac_paths:
    flac_file = os.path.basename(i)
    speaker_id = flac_file.split('_')[0]
    check_dir(base_dir + target_dir + speaker_id)
    if flac_file.split('.')[0] not in transcript_dict:
        print(f"Transcript not found for {flac_file}")
        continue
    os.system('cp ' + i + ' ' + base_dir + target_dir + speaker_id + '/' + flac_file)
    # write the transcript to a file
    with open(base_dir + target_dir + speaker_id + '/' + flac_file.split('.')[0] + '.txt','w') as f:
        f.write(' '.join(transcript_dict[flac_file.split('.')[0]]))
    count += 1
    if count%100 == 0:
        print(count)
    if count == max_count:
        break

