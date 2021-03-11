# Lyrics Emotion Classifier
1. Requirements:
    - You need to have python installed.
    - Command for installing dependencies
        -`pip install -r requirements.txt`, or `pip3 install -r requirements.txt`

2. Instructions to Run the code:
    - Expected to provide the path to moody lyrics dataset or any required dataset in line 102, assign list of lyrics to the variable songs in line 106 and numeral mood categories in line 142. 
    - `python3 --lr lr --epochs epochs --ml max_len --bs batch_size --ts test_size --adaptive alr run.py`
    where,
        - lr is the learning rate(in the range of e-5, 2e-5) 
        - epochs is number of epochs
        - max_len is max length(in the range of 512, 1024)
        - batch_size is batch size(depends on your computing power)
        - test_size is test size in fractions(Eg:- 0.1,0.2)
        - alr is denominator in adaptive learning rate function(in the range of 20 - 40)
            - for given alr = 40, the adaptive rate function would be `f(lr,ith epoch) = lr*(0.1)^(i/40)`
    - All the statements will be printed in a file in logs directory named according to the parameters(xlnet_maxlen_bs_batchsize_adamw_data_trainsize_lr_learningrate_adaptiveratedenominator).

Eg: `python3 run.py --lr 2e-5 --epochs 10 --ml 1024 --bs 8 --ts 0.2 --adaptive 20`
will generate a log file named `xlnet_1024_bs_8_adamw_data_80_lr_2e5_20` in the ../logs directory.
