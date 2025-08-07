An NLP sentiment analysis system that classifies Reddit comments into a curated set of high-signal emotions using a fine-tuned DistilBERT transformer. This project emphasizes dataset quality over model complexity, applying advanced noise reduction and label engineering before training to ensure highly discriminative learning.

Key Features Implemented So Far
Curated Dataset – Filtered the 27-label GoEmotions dataset to a smaller, domain-relevant set of emotions.

Noise Reduction – Removed multi-label samples and low-signal emotions to improve class separability.

Custom Label Remapping – Mapped GoEmotions labels to a compact, contiguous index space for efficiency.

Transformer-Ready Tokenization – Used Hugging Face’s AutoTokenizer to prepare text for DistilBERT embeddings.

Next Steps
Fine-tune DistilBERT using PyTorch Lightning.

Evaluate performance per emotion with F1-score, precision, and recall.

Implement class imbalance handling strategies.
