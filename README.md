# Speech_Emotion_Recognition
SER using cepstral features: MFCC, LFCC, GFCC

Dataset used: RAVDESS
DL Classifier used: CNN, BiLSTM
<br>
<br>
1. **Mel Frequency Cepstral Coefficients (MFCC)**

Description: MFCC replicates human auditory perception by applying the Mel scale, which is non-linearly spaced—denser at lower frequencies and more spread out at higher frequencies.
Key Features:
Extracts vocal tract characteristics using a logarithmic filterbank.
Emphasizes speech phonetics over pitch information.


Applications: Widely used in speech recognition and audio analysis.
<br>
<br>
2. **Linear Frequency Cepstral Coefficients (LFCC)**

Description: LFCC employs linearly spaced filters, providing equal emphasis on low and high-frequency components, unlike the Mel scale.
Key Features:
Captures detailed spectral information based on raw signal characteristics.
Less affected by nonlinear distortions in speech.
Effective for analyzing subtle spectral variations.


Applications: Particularly beneficial for medical and diagnostic applications, such as detecting abnormalities.
<br>
<br>
3. **Gammatone Frequency Cepstral Coefficients (GFCC)**

Description: GFCC uses gammatone filterbanks that mimic the human cochlear response, capturing both formant and pitch information.
Key Features:
Robust to noise, making it ideal for challenging acoustic environments.
Excels at identifying fine-grained spectral details.


Applications: Highly effective for speech enhancement, speaker recognition, and other tasks in noisy conditions.

