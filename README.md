# Verb Form Prediction using BERT

This project leverages the power of BERT (Bidirectional Encoder Representations from Transformers) for predicting verb forms. BERT, a state-of-the-art transformer-based model developed by Google, has shown remarkable performance in various natural language processing tasks, including text classification.

### Introduction

Predicting verb forms is crucial in understanding the syntactic structure and grammatical nuances of sentences. This project focuses on predicting whether a given verb form should be in singular ('VBZ') or plural ('VBP') based on the context provided in the sentence.

### Setup and Installation

To run this project, ensure you have the necessary Python libraries installed. You can install them using pip:

```bash
!pip install accelerate
!pip install transformers torch scikit-learn pandas
```

### Data Preparation

The project utilizes a dataset containing verb forms annotated with their respective parts of speech (POS) tags. This dataset is split into training and testing sets for model evaluation.

### Tokenization and Dataset Creation

The text data is tokenized using BERT's tokenizer, preparing it for input into the BERT model. Custom PyTorch datasets are created to efficiently handle the tokenized inputs and labels.

### Model Training

A BERT model for sequence classification is initialized and trained using the training dataset. The model is fine-tuned to predict the verb forms accurately.

### Verb Prediction Function

A function is defined to predict the verb form of a given sentence using the trained BERT model. This function processes the input sentence through the model and outputs the predicted verb form ('VBZ' for singular, 'VBP' for plural).

### Evaluation and Metrics

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Training metrics like loss, gradient norms, and learning rate are also monitored and visualized to assess training dynamics.

### Save and Deployment

Finally, the trained model and tokenizer are saved for future use or deployment in production environments.

This project provides a robust framework for verb form prediction using BERT, demonstrating its effectiveness in handling grammatical tasks in natural language processing.
