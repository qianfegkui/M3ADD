# Dependencies
python==3.9.7
torch==1.13.1+cu122
torch-geometric== 2.5.3
matplotlib==3.6.2
scipy==1.10.0
numpy==1.24.1 
seaborn==0.13.2 

## Dataset
To be announced upon acceptance of the paper.

## Training networks 

    python run.py 


## Load the saved model.
    python loadmodel.py   

## Performance 

| Models        |                            | | |Depression | |       |                         | |Wellbeing | | |             |
|---------------|---------|---------|---------|---------|---------|---------|----------|---------|---------|---------|---------|---------|
|               |   EEG   ||   EMG |  | EEG+EMG ||   EEG   ||   EMG   || EEG+EMG  ||
|               |  Acc.   |   F1    |  Acc.   |   F1    |  Acc.   |   F1    |  Acc.    |   F1    |  Acc.   |   F1    |  Acc.   |   F1    |
| M³ADD | **81.58**|**81.42**|**73.68**|**73.47**|**86.84**|**87.09**|**90.24**|**90.07**|**85.37**|**85.39**|**95.12**|**95.26**|

