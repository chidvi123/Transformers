# Hallucination Reduction in LLMs

## 1. What is AI Hallucination?

AI hallucination occurs when an AI model generates information that sounds convincing but is incorrect, fabricated, or unsupported by reliable evidence.

**Example:** An LLM confidently provides a fake research paper or an incorrect company policy.

## 2. Why Do LLMs Hallucinate?

- **Lack of Knowledge:** The model doesn't have sufficient information.
- **Outdated Information:** The model may rely on old information.
- **Ambiguous Prompts:** Unclear questions can lead to incorrect assumptions.
- **Insufficient Context:** Missing or irrelevant information can result in unsupported answers.
- **Probabilistic Generation:** LLMs predict likely next tokens and don't automatically verify factual correctness.

## 3. Techniques to Reduce Hallucinations

### 1. Retrieval-Augmented Generation (RAG)
Retrieves relevant information from external sources and provides it to the LLM as context.

**Example:** Retrieving the latest HR policy before answering an employee's question.

### 2. Prompt Engineering
Providing clear instructions to guide the model towards accurate and relevant responses.

**Example:** Instructing the model to answer only using the provided context and avoid assumptions.

### 3. Grounding with Reliable Sources
Connecting generated responses to trustworthy information.

**Example:** Using verified medical documents to answer healthcare-related questions.

### 4. Restricting Answers
Allowing the model to acknowledge when information is unavailable instead of generating guesses.

**Example:** Responding with "I couldn't find this information in the available records."

### 5. Human Verification
Having humans review important AI-generated outputs before they are used.

**Example:** A healthcare professional reviewing an AI-generated medical report summary.

## 4. Combining Techniques

We can combine multiple techniques to reduce hallucinations.

**Example: Company AI Chatbot**

1. RAG retrieves the latest company policies.
2. Prompt Engineering instructs the LLM to answer using retrieved information.
3. Grounding ensures responses are supported by the retrieved documents.
4. Answer Restrictions prevent the model from guessing when information is missing.
5. Human Verification reviews sensitive or high-impact responses.

**Result:** A more reliable chatbot with a lower risk of hallucinations.