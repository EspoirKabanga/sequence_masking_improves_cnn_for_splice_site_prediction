# Sequence Masking Improves CNNs for Splice Splice Prediction

This repository contains the codes for the experimental work presented in the following paper:
**Strategic Sequence Masking Improves CNN Effectiveness in Splice Site Prediction**

## Abstract

This study investigates the effectiveness of convolutional neural networks (CNNs) in predicting splice sites, focusing on the role of sequence masking and the variability of splice site positioning. We implement three types of masking on the data: upstream masking, where masking is applied from the upstream edge of the sequence toward the splice site; downstream masking, which is applied from the downstream edge of the sequence toward the splice site; and bidirectional masking, applied from both edges of the sequence toward the splice site. We progressively increase the masked regions, starting from 15%, 30%, 45%, 60%, 75%, and up to 90% of the upstream or downstream region length, with the 90% masking being very close to the splice site. Additionally, we experiment with the positional variability of splice sites by randomly repositioning them within the sequences, diverging from traditional fixed positions. Our findings indicate that the masking strategy substantially improves the effectiveness of CNNs in splice site prediction. However, when splice sites are randomly repositioned within DNA sequences, the performance of CNNs decreases notably. These results demonstrate the importance of strategic sequence modifications in optimizing CNN-based splice site prediction.

## Data

the dataset used in the work can be downloaded through the following link:
https://zenodo.org/records/14001778

## Requirements

\>= Python 3.6

\>= Tensorflow 2.10.1

\>= numpy 1.24.3
