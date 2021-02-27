# Lyrics Emotion Classifier
1. Requirements:
    - You need to have python installed.
    - Command to installed dependencies
        -`pip install -r requirements.txt (Python 2)`, or `pip3 install -r requirements.txt`

2. Instructions to Run the code:
    - Expected to provide the path to moody lyrics dataset or any required dataset in line 50 and assign lyrics to the variable songs in line 111. 
    - `python3 --lr x --ne y --ml z --bs b --ts t --denom d`
    where x is the learning rate(in the range of e-5), y is number of epochs(< 20), z is max lenght(2^9, 2^10), b is batch size(depends on your computing power), t is test size and d is denominator in adaptive learning rate function(20-40).
    - All the statements will be printed in a file in logs according to the parameters
