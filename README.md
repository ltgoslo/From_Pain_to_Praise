# From Pain to Praise: Aspect-Based Sentiment-Analysis for Norwegain Patient Feedback
This repository contains the code, link to models and ABSA guidelines associated with our paper.

### Models
#### NorBERT3 large
We publish four fine-tuned versions of the NorBERT3 large model:
- Sentence-level, fine-grained aspects, fine-tuned on the train split
- Sentence-level, coarse-grained aspects, fine-tuned on the train split
- Sentence-level, fine-grained aspects, fine-tuned on the full dataset
- Sentence-level, coarse-grained aspects, fine-tuned on the full dataset

The two latter models are naturally not evaluated, but represent what will be used in practice by [NIPH](https://www.fhi.no/en/)/[FHI](https://www.fhi.no/) (The Norwegian Institute of Public Health/Folkehelseinstituttet).

#### NorMistral-11b-thinking
Although the encoder model is clearly the better choice for this task, we publish one fine-tuned version of NorMistral-11b-thinking; normistral-absa, which is fine-tuned on sentence-level texts and fine-grained aspects.

### ABSA Guidelines
You can find the guidelines used for ABSA annotation in this repository. The pdf contains descriptions of each aspect, annotation examples and overall annotation guidelines.

### Citation
Coming soon.
