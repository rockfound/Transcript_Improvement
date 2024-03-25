RF Transcript Improvement is a tool that can be used to improve speaker identification of full screen video call transcripts.  This done by analysing a single frame from each utterance using facial recognition software to identify the speaker.  If a single speaker is attributed to more than one sequential utterance each utterance in the sequence is combined into a single utterance.  Given a transcript and video rfti.py will generate a `.csv` file containing the new speaker lables, new utterance texts, and new timestamps.


# Requirements
1. {transcript to fix}.csv: A transcript from a fullscreen video call.  A full screen video call is a video call where only the current speaker’s camera is being shown.  
The {transcript to fix}.csv file should contain following columns:
    - ut_id: a unique id for each utterance
    - timestamp: the timestamp for each utterance of the from H:M:S
    - text: the utterance text
1. a video recording of the transcript. 

# Environment set up
```sh
conda create -n {environment name} python=3.11
conda activate {environment name}
conda install pandas 
conda install jupyter
```
## TensorFlow GPU

```sh
nvidia-smi # check to make sure the NVIDIA GPU Driver is installed
python3 -m pip install tensorflow[and-cuda]
# Verify TensorFlow runs on gpu:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
In some cases, you will be required to specify which version of TensorFlow to install.  More information can be found here: 
- https://www.tensorflow.org/install/pip 
- https://www.tensorflow.org/install/source#tested_build_configurations
## TensorFlow CPU 
```sh
pip install tensorflow
```
## Install DeepFace
```sh
pip3 install deepface 
```

# Run from Command Line
Navigate to the `transcript_improvement` directory and execute the following command
```sh
python .rfti/rfti.py {transcript file name} {path to video} {output file name}
```
# Output
rfti.py outputs  `{output filename}.csv` with the following collumns
- ut_id(int): the new utterance id 
- text(string): text for each utterance
- speaker(int): the new speaker id for each utterance
- timestamp(string): the new timestamps in the format Hour:Minute:Second
- combined(bool): wether or not an utterance was combined with other utterances
- original_ut: the original utterance id or the utterance id of the first utterance combined
# Jupyter Notebook
If you would like to run the transcript improvement pipeline in an interactive notebook you can update the variables in the 2nd cell in the transcript_improvement.ipynb jupyter notebook


# More details
[Transcript Improvement Blog](https:google.com)

