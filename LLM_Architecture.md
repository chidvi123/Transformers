# BERT and GPT Architecture

# 1. BERT (Bidirectional Encoder Representations from Transformers)

## What is BERT?

BERT is an **Encoder-only Transformer** designed mainly for understanding and representing text.

It uses **Bidirectional Self-Attention**, meaning each token can consider both the left and right context.

Example:

`The bank is beside the river`

BERT uses the surrounding context to understand that `bank` refers to a riverbank rather than a financial institution.

## BERT Architecture

![BERT Transformer Encoder Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20261001161109102317/transformer_encoder.webp)

*Source: GeeksforGeeks – [BERT Model](https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/)*

Original Research Paper: [BERT – Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Working of BERT

### Step 1: Tokenization

The input sentence is converted into smaller tokens and then mapped to token IDs.

Example:

`The movie was amazing`

Tokens: `[The, movie, was, amazing]`

### Step 2: Input Embeddings

Each token ID is converted into a dense vector representation.

For example, if the embedding dimension is 768, each token is represented by a 768-dimensional vector.

### Step 3: Positional Embeddings

Positional embeddings provide information about the position of each token in the sentence.

### Step 4: Segment Embeddings

Segment embeddings identify which sentence a token belongs to. They are useful when processing sentence pairs.

The final input representation is obtained by combining:

- Token Embeddings
- Positional Embeddings
- Segment Embeddings

### Step 5: Transformer Encoder

The input representations are passed through multiple Transformer Encoder layers.

Each Encoder layer contains:

1. Multi-Head Self-Attention (Bidirectional).
2. Residual Connection + Layer Normalization.
3. Feed Forward Network.
4. Residual Connection + Layer Normalization.

The Encoder produces contextualized representations for every input token.

### Step 6: Task-Specific Output Layer

The Encoder outputs are used for different NLP tasks, such as classification, NER and question answering.

## BERT for Classification

For sentence classification, the final representation of the special `[CLS]` token is commonly passed to a Linear layer.

**Architecture flow:**

`Input → Tokenization → Embeddings → BERT Encoder → [CLS] Representation → Linear Layer → Softmax → Class Prediction`

### Example: Sentiment Classification

Suppose the input is:

`The movie was amazing`

The classification layer produces logits for the available classes.

Softmax converts the logits into probabilities.

| Class | Probability |
|---|---:|
| Positive | 88% |
| Negative | 12% |

Final Prediction: **Positive**

Softmax is generally used for mutually exclusive classes. For multi-label classification, independent Sigmoid outputs are commonly used.

## BERT Pre-training Objectives

The original BERT model uses two training objectives:

### 1. Masked Language Modeling (MLM)

Some tokens in the input sentence are masked, and the model learns to predict the masked tokens using surrounding context.

Example:

`The movie was [MASK]`

Prediction: `amazing`

### 2. Next Sentence Prediction (NSP)

The model learns to predict whether one sentence follows another in the original text.

Note: NSP was part of the original BERT training setup, but not every later BERT-style model uses it.

## Applications of BERT

- Sentiment Analysis
- Text Classification
- Named Entity Recognition (NER)
- Question Answering
- Search Relevance

---

# 2. GPT (Generative Pre-trained Transformer)

## What is GPT?

GPT is a **Decoder-only Transformer** designed primarily for autoregressive text generation.

It predicts the next token based on the tokens that came before it.

Example:

Input: `I love machine`

Possible next token: `learning`

GPT then uses the extended sequence to predict the next token.

## GPT Architecture

![GPT Decoder Only Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20260324090208535998/Decoder-only.webp)

*Source: GeeksforGeeks – [Introduction to GPT](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-generative-pre-trained-transformer-gpt/)*

Original Research Paper: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## Working of GPT

### Step 1: Input Prompt

The model receives the user's input text.

Example:

`Explain machine learning`

### Step 2: Tokenization

The input text is converted into token IDs.

A token can represent:
- A complete word.
- Part of a word.
- Punctuation.

### Step 3: Token Embeddings

Each token ID is converted into a dense vector representation.

For example, assume the embedding dimension is 512.

If the input contains 3 tokens:

\[
X = 3 \times 512
\]

### Step 4: Positional Information

Positional information is added to the token representations so that GPT understands the order of tokens.

### Step 5: Transformer Decoder Blocks

The representations pass through multiple Transformer Decoder blocks.

Each Decoder block contains:

1. Masked Multi-Head Self-Attention (Causal Attention).
2. Residual Connection + Layer Normalization.
3. Feed Forward Network.
4. Residual Connection + Layer Normalization.

Unlike the original Encoder-Decoder Transformer, GPT does not have a separate Encoder.

Therefore, it does not require Cross-Attention.

Its Query, Key and Value representations are generated from the Decoder's own input representations.

## Why Does GPT Use Masked Self-Attention?

GPT must not look at future tokens while predicting the next token.

Example:

`I love machine learning`

| Current Token | Tokens it can attend to |
|---|---|
| I | I |
| love | I, love |
| machine | I, love, machine |
| learning | I, love, machine, learning |

The causal mask prevents a token from attending to future positions.

**Important:** Causal masking is used during both training and inference.

## Step 6: Linear Layer

After passing through all Decoder blocks, the final hidden representation is passed through a Linear layer.

Suppose:

- Hidden dimension = 512
- Vocabulary size = 50,000

For the final input position:

\[
(1 \times 512)(512 \times 50000)
\]

\[
= 1 \times 50000
\]

The Linear layer produces 50,000 logits, one for each token in the vocabulary.

## Step 7: Softmax

Softmax converts the logits into probabilities for possible next tokens.

\[
P(token_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Example:

| Possible Next Token | Probability |
|---|---:|
| learning | 60% |
| coding | 20% |
| today | 10% |
| Others | 10% |

The decoding strategy selects a **token**, not a probability.

The probability indicates how likely the model considers that token.

## Step 8: Token Selection

GPT selects a token using a decoding strategy.

Common strategies:

- **Greedy Decoding:** Selects the token with the highest probability.
- **Sampling:** Selects a token based on its probability distribution.
- **Temperature:** Controls how concentrated or diverse the probability distribution is.

The selected token ID is converted into its token representation.

## Step 9: Autoregressive Generation

The selected token is appended to the existing sequence.

Example:

Initial input:

`I am`

| Current Sequence | Predicted Next Token |
|---|---|
| I am | a |
| I am a | boy |
| I am a boy | EOS |

The sequence grows one token at a time until generation is complete.

### Important: Embedding and Positional Information During Generation

Whenever a new token is generated:

1. Its token ID is obtained.
2. Its embedding is retrieved.
3. Positional information is added for its position.
4. It is appended to the existing sequence.
5. The Decoder predicts the next token.

The sequence shape grows as new tokens are added.

For example, assuming an embedding dimension of 512:

| Sequence | Shape |
|---|---:|
| I am | 2 × 512 |
| I am a | 3 × 512 |
| I am a boy | 4 × 512 |

In practical implementations, previous Key and Value representations are commonly stored using a **KV Cache**, avoiding unnecessary recomputation.

**Important:** GPT does not necessarily start with a `[BOS]` token. In normal chat or completion use, the user's prompt is the initial sequence. Some model formats also use special beginning-of-sequence or message-control tokens.

## GPT Training vs Inference

| Training | Inference |
|---|---|
| Learns to predict the next token in training sequences | Generates tokens from a user prompt |
| Uses causal masking | Uses causal masking |
| Compares predictions with actual next tokens | Selects the next token using a decoding strategy |
| Updates weights using loss and backpropagation | Model weights generally remain fixed |

During training, predictions for multiple positions can be calculated in parallel while causal masking prevents access to future tokens.

During inference, tokens are generated sequentially.

---

# 3. BERT vs GPT

| Feature | BERT | GPT |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional Self-Attention | Causal Self-Attention |
| Main Purpose | Text Understanding | Text Generation |
| Original Training Objective | MLM + NSP | Next Token Prediction |
| Typical Output | Contextualized Token Representations | Next Token Probabilities |
| Cross-Attention | Not used | Not used in standard Decoder-only GPT |
| Common Applications | Classification, NER, QA | Chatbots, Text Generation, Summarization |

---

# 4. Quick Recap

## BERT

- Uses an Encoder-only architecture.
- Uses Bidirectional Self-Attention.
- Mainly used for understanding and classification tasks.
- Produces contextual representations for input tokens.
- Classification flow:

`Input → BERT Encoder → [CLS] → Linear → Softmax → Class`

## GPT

- Uses a Decoder-only architecture.
- Uses Causal Self-Attention.
- Mainly used for text generation.
- Predicts one token at a time.
- Generation flow:

`Prompt → Tokenization → Embeddings → Decoder Blocks → Linear → Softmax → Token Selection → Append Token → Repeat`

---

# 5. References

### Published Architecture Sources

1. [GeeksforGeeks – BERT Model](https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/)
2. [GeeksforGeeks – Introduction to GPT](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-generative-pre-trained-transformer-gpt/)

### Original Research Papers

1. [BERT – Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)](https://arxiv.org/abs/1810.04805)
2. [GPT – Improving Language Understanding by Generative Pre-Training (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
3. [Transformer – Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

*Architecture images are embedded from the published GeeksforGeeks image sources. Please retain the source attribution when reusing them.*