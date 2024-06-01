# Fine-Tuning GPT-2 for Classification with LoRA and Knowledge Distillation

In this project, I aim to fine-tune the GPT-2 and GPT-2 Medium models for binary classification tasks using two techniques: LoRA (Layer-wise Relevance Analysis) and Knowledge Distillation.

## Overview

I utilize two variants of the pre-trained language model (LM) GPT-2:

- GPT-2: 125.11 million parameters
- GPT-2 Medium: 356.62 million parameters

Both of these models are being fine-tuned using different approaches for the COLA dataset. The COLA dataset is designed for binary classification, specifically determining whether a given sentence is grammatically correct or not.

The project is divided into two parts:

1. **LoRA Fine-Tuning**: In the first part, I use the LoRA technique to efficiently fine-tune the GPT-2 and GPT-2 Medium models. LoRA is a method that allows us to adapt large pre-trained models to new tasks with a small amount of task-specific parameters and data.

2. **Knowledge Distillation**: In the second part, I use Knowledge Distillation to train a smaller RNN model for classification. Knowledge Distillation is a technique where a smaller (student) model is trained to mimic the behavior of the larger (teacher) model, in this case, the fine-tuned GPT-2 or GPT-2 Medium model. This results in a smaller, more efficient model that maintains a high level of performance.


To know more details about the project check the pdf provided and Assignmnet2 contain all the code for the project.
