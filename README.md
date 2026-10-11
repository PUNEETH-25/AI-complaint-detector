HACKNETIX - Fine-Tuned Sentence Transformer

CONTENTS
models/hacknetix_sentence_transformer/
    Fine-tuned Sentence Transformers model weights, tokenizer, and configuration files.

BACKEND LOADING
The Flask backend loads this model by default. Its absolute path is derived from
backend/config.py, so launching Flask does not depend on the terminal's current directory.
Install the backend dependencies with `pip install -r backend/requirements.txt`.

To load it directly from a Python file in the project root:

from sentence_transformers import SentenceTransformer
from pathlib import Path

project_root = Path(__file__).resolve().parent
model_dir = project_root / "ai_model" / "models" / "hacknetix_sentence_transformer"
model = SentenceTransformer(str(model_dir))

SIMILARITY EXAMPLE
from sentence_transformers import util

embeddings = model.encode(
    [complaint_a, complaint_b],
    convert_to_tensor=True,
    normalize_embeddings=True
)

similarity = util.cos_sim(
    embeddings[0], embeddings[1]
).item()

IMPORTANT
- This model is fine-tuned from all-MiniLM-L6-v2; it is not the unchanged base model.
- Similarity scores are not probabilities.
- The application's existing 0.60 suggestion cutoff has not been calibrated for this model.
- Calibrate a duplicate-detection threshold using held-out complaint data before relying on it.
- The model does not itself provide Flask routes, a database, or a
  complete complaint-grouping system.
- Verify compatibility with the sentence-transformers version
  installed in your backend environment.
