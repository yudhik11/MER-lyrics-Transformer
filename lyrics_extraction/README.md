# Lyrics Extraction
1. Lyric_Extraction_GoogleSearch.ipynb 
    - This function uses google search to get the required genius url to scrape the lyrics.
    - This handles spelling mistakes in artist and track names.
    - There is a limit on google search limit after which the code terminates, use it while extracting fewer songs without multiprocessing.

2. Lyric_Extraction_WithoutGoogleSearch.ipynb
    - This function uses generates the genius url for the song by doing some minimal preprocessing.
    - This doesn't handle spelling mistakes in artist and track names.
    - There are no request limit so use this funciton to extract lyrics for large number of songs, can be used with multiprocessing.
