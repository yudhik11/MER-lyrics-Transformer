# Lyrics Extraction
1. Lyric_Extraction_GoogleSearch.ipynb 
    - This function uses google search to get the required genius url to scrape the lyrics.
    - This handles spelling mistakes in artist and track names.
    - There is a limit on google search from an ip address after which the code terminates, use it while extracting fewer songs and without multiprocessing.
    - Instruction to run: Extract(Trackname, Artist Name).
    - Eg:- Extract("Shap of U", 'Ed')
    - Execution time for one valid song: 6.96sec

2. Lyric_Extraction_WithoutGoogleSearch.ipynb
    - This function generates the genius url for the song by doing some minimal preprocessing on artist and track names.
    - This doesn't handle spelling mistakes in artist and track names.
    - There are no request limit so use this funciton to extract lyrics for large number of songs, also can be used with multiprocessing.
    - Instruction to run: Extract(Trackname, Artist Name).
    Eg:- Extract('Shape Of You', 'Ed Sheeran')
    - Execution time for one valid song: 5.43sec
