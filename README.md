# CS182 Final Project: Yelp Review Classifier

## Submission Details:

### Final Submission:
To run our final submission, download Best_Submission_BERT_Ordinal.zip from https://drive.google.com/open?id=1VVoluGea4sh6HpjfdcWjlzC82DE5oZuL, and run it in the way specified in the spec, i.e. unzip and then python test_submission.py <input.jsonl>.

### Other Models:
To run our best RNN model, download rnn-submission.zip from https://drive.google.com/open?id=1VVoluGea4sh6HpjfdcWjlzC82DE5oZuL, and run it in the way sepecfied in the spec, i.e. unzip and then python test_submission.py <input.jsonl>.

To run our BERT model without ordinal labels, download Archive.zip from https://drive.google.com/drive/u/0/folders/11jn8qVLVRMjwmv-REYTimGLYdsn76nAs, and run it in the way sepecfied in the spec, i.e. unzip and then python test_submission.py <input.jsonl>.

## Code Structure:
### RNN Folder:
The Main Data For This Section Can Be Accessed In The CSV At https://drive.google.com/open?id=1VVoluGea4sh6HpjfdcWjlzC82DE5oZuL. Requirements For This Section Can Be Found In The rnn-submission.zip Mentioned Above.

- BoW.ipynb - Notebook for our Bag of Words Baseline Model
- Challenge Sets.ipynb - Notebook For Evaluation Of The Challenge Sets Released By Staff
- Clean Data.ipynb - Notebook To Test Data Cleaning Methods
- Testing/Testing2/Testing3/Testing4.ipynb - Notebook For Initial Experimentation
- Evaluation Book.ipynb - Notebook For Model Evaluation
- Finetuning.ipynb - Notebook For Finetuning Experimentation And Non-Sequential Model Experimentation
- RNN Classifier.ipynb - Basic RNN Classifier Notebook
- Tree Model.ipynb - Notebook For Tree Model Experimenation
- models.py - File With Model Structures For Different Model Types
- perturbations.txt - Text File For Thesaurization
- tokenizer.txt - Our Pretrained Tokenizer

### BERT Folder
The Main Data For This Section Can Be Accessed By Converting Our CSV Data To A JSON Format, We Did Not Include This As The File Was Large And We Felt Like Even In The Abscence Of The Specific Data File, The Code Speaks For Itself. Requirements For This Section Can Be Found In The Best_Submission_BERT_Ordinal.zip Mentioned Above.

- train_binary/train_binary1.py - Binary Classification BERT Models
- train_crossentropy.py - Base BERT With Regular Loss Function
- train_modifiedentropy.py - BERT With Modified Loss Function
- train_ordinal.py - BERT With Loss Function Based On Ordinal Labels

### Additional Models Folder
The Main Data For This Section Is In The ultra_clean.csv File. Requirements For This Section Can Be Found In The requirements.txt File.

- requirements.txt - Requrements For This Section
- word2vec.ipynb - Notebook For Our Word2Vec Model: The Model Can Be Found At https://drive.google.com/drive/u/1/folders/1zQ6ZVp2MVCxvlFQC5n9R2qg3DCSIumIW.
- CNNLSTM.ipynb - Notebook For Our CNN+LSTM Model
- ultra_clean.csv - Data For This Section
