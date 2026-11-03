# LLM Evaluation

## 1. What is LLM Evaluation?

LLM Evaluation is the process of measuring how well a Large Language Model performs in terms of accuracy, relevance, reliability, and quality of generated responses.

**Example:** Evaluating whether a customer-support chatbot provides correct return policies, relevant answers, and reliable information.

## 2. Why Do We Evaluate LLMs?

- **Accuracy:** Check whether the generated information is correct.
- **Relevance:** Ensure answers match the user's question.
- **Reliability:** Check whether the model consistently produces good responses.
- **Safety:** Identify harmful or inappropriate outputs.
- **Performance:** Understand response time and computational cost.

## 3. Traditional Evaluation Metrics

### 1. Accuracy
Measures how many predictions are correct out of the total predictions.

$$
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
$$

### 2. Precision, Recall and F1-score

Useful for evaluating classification and information-extraction tasks.

- **Precision:** Out of all positive predictions, how many were actually correct?
- **Recall:** Out of all actual positive cases, how many did the model identify?
- **F1-score:** Harmonic mean of precision and recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### 3. BLEU Score

BLEU (Bilingual Evaluation Understudy) compares generated text with reference text using matching word sequences (n-grams).

- Precision-oriented metric.
- Checks how much of the generated text matches the reference.
- Commonly used in machine translation.
- Uses a brevity penalty to penalize excessively short outputs.

### 4. ROUGE Score

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures how much reference content is captured by generated text.

- Recall-oriented metric.
- Checks how much of the reference content appears in the generated output.
- Commonly used in text summarization.

### BLEU vs ROUGE

| BLEU | ROUGE |
|---|---|
| Precision-oriented | Recall-oriented |
| Focuses on generated text matching the reference | Focuses on reference content captured by the output |
| Commonly used in translation | Commonly used in summarization |

**Note:** BLEU and ROUGE measure text overlap, not necessarily the actual semantic meaning.

### 5. Perplexity

Perplexity measures how well a language model predicts a sequence of tokens.

- Lower perplexity generally indicates better predictive performance.
- Higher perplexity indicates greater uncertainty when predicting tokens.
- Commonly used to evaluate language models.

**Important:** Lower perplexity does not necessarily mean better factual accuracy.

## 4. LLM-specific Evaluation Metrics

These metrics are particularly useful when evaluating RAG-based applications.

### 1. Faithfulness

Measures whether the generated answer is supported by the retrieved context.

**Example:** If the retrieved document says employees receive 20 annual leaves, the LLM should not generate 30 days.

### 2. Answer Relevance

Measures whether the generated answer actually addresses the user's question.

**Example:** If the user asks about annual leave entitlement, the answer should directly explain the number of available leave days.

### 3. Contextual Precision

Measures whether the most relevant information is ranked higher in the retrieved results.

**Example:** When searching for annual leave policies, relevant leave documents should appear before unrelated insurance documents.

### 4. Contextual Recall

Measures whether the retrieval system successfully retrieves all relevant information needed to answer the question.

**Example:** If a question asks about annual leave entitlement and carry-forward rules, the retriever should retrieve both relevant policies.

### Quick Revision

| Metric | What does it check? |
|---|---|
| Faithfulness | Is the answer supported by retrieved context? |
| Answer Relevance | Does the answer address the user's question? |
| Contextual Precision | Are relevant documents ranked higher? |
| Contextual Recall | Are all required relevant documents retrieved? |

## 5. Who Evaluates These Metrics?

LLM evaluation can be performed using automated evaluation frameworks and human reviewers.

### A. Automated Evaluation

Developers use evaluation frameworks to automate the evaluation process.

**Popular tools:**

- **Ragas:** Evaluates RAG applications using metrics such as faithfulness, answer relevancy, context precision and context recall.
- **DeepEval:** Provides metrics and testing capabilities for LLM applications.
- **LangSmith:** Helps evaluate, trace, and monitor LLM applications.

These tools can calculate evaluation scores and help identify weaknesses in the application.

### B. Human Evaluation

Human reviewers manually examine:

- Whether answers are factually correct.
- Whether answers are useful and relevant.
- Whether the model produces misleading information.
- Whether automated evaluation results are reliable.

### Evaluation Workflow

1. LLM generates a response.
2. Automated evaluation frameworks calculate relevant metrics.
3. Human reviewers validate important results.
4. Developers identify weaknesses and improve the application.

**Important:** Automated evaluation does not guarantee correctness. An LLM acting as an evaluator can also make mistakes, so human evaluation remains important.