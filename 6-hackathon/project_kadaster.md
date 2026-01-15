### AI Challenge - Kadaster

The Kadaster registers over 100 types of legal events (rechtsfeiten), such as mortgages, seizures, or sales. However, recognizing these automatically is a major challenge because notaries use unstructured text without a fixed format. While standard models successfully identify common events, they fail on the Long Tail rare legal facts that occur infrequently (e.g., fewer than 20 times), making them impossible to learn via traditional training.

The Kadaster dataset suffers from a "Long Tail" distribution. While standard models (Regex/Neural) handle common "rechtsfeiten" well, they fail on rare legal facts where training data is scarce (fewer than 20 examples).

### Solution
To solve this, Christel and I developed a Zero-Shot Learning approach specifically for these rare cases. instead of training on examples, we used Large Language Models (LLMs) to recognize facts based on their legal descriptions.

### Methodology & Achievements

We benchmarked 5 different models and identified the top two performers based on context handling and accuracy:

- Qwen/Qwen3-30B-A3B-Thinking (262k context window)

- OpenAI/GPT-OSS-120B (131k context window)
  

- Advanced prompt engineering: we significantly improved performance by refining the prompt structure:

- Context injection: enriched the prompt with the rechtsfeiten definitions and added specific rules (e.g., synonyms).

- Noise reduction: instructed the model to ignore unnecessary historical information within the deeds.

- Hallucination prevention: built in an "escape mechanism" to ensure the model does not invent facts if the confidence is low.

- Configuration: we configured the system to target the long tail (threshold < 20 occurrences) with a context limit of 30,000 tokens to balance performance and speed.

### Results 
We successfully established a workflow that allows the system to recognize rare legal facts that the main model misses, with the Qwen and OpenAI models achieving comparable F1 scores (Micro F1 ~0.66).

You can find all the code in (https://github.com/christelvanharen/MADS_hackaton)

[Go back to Homepage](../README.md)
