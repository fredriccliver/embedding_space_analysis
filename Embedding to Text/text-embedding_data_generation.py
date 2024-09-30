# prepare more data using openai embedding model
# in ./data/episodes.json, the embedding is actually corresponding to the summary, not the title.
# to finetune gpt2 base model, prepare 5,000 (min) data (title, embedding pairs)
# embedding dimension: 1536 (text-embedding-3-small)
# and then, use this new dataset to finetune gpt2 base model (to make Embedding to Text model)
