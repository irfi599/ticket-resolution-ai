from sentence_transformers import SentenceTransformer

print("Starting embedding test...")

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

text = "My calls keep dropping and my mobile internet is very slow."

print("\nCreating embedding...")

embedding = model.encode(text)

print("Embedding created successfully!")

print("Embedding type:", type(embedding))
print("Embedding size:", embedding.shape)

print("\nTest completed successfully!")