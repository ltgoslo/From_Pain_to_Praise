# From Pain to Praise: Aspect-Based Sentiment-Analysis for Norwegain Patient Feedback
This repository contains the full paper, code, link to models and ABSA guidelines associated with our paper.

### Models
#### NorBERT3 large
We publish four fine-tuned versions of the NorBERT3 large model:
- [norbert3-fine-absa](https://huggingface.co/ltg/norbert3-fine-absa): Sentence-level, fine-grained aspects, fine-tuned on the train split
- [norbert3-coarse-absa](https://huggingface.co/ltg/norbert3-coarse-absa): Sentence-level, coarse-grained aspects, fine-tuned on the train split
- [norbert3-fine-absa-full](https://huggingface.co/ltg/norbert3-fine-absa-full): Sentence-level, fine-grained aspects, fine-tuned on the full dataset
- [norbert3-coarse-absa-full](https://huggingface.co/ltg/norbert3-coarse-absa-full): Sentence-level, coarse-grained aspects, fine-tuned on the full dataset

The two latter models are not evaluated, but represent what will be used in practice by [NIPH](https://www.fhi.no/en/)/[FHI](https://www.fhi.no/) (The Norwegian Institute of Public Health/Folkehelseinstituttet).

### ABSA Guidelines
You can find the guidelines used for ABSA annotation in this repository. The pdf contains descriptions of each aspect, annotation examples and overall annotation guidelines.

### Citation
Coming soon.
