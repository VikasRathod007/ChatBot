from transformers import (
    BertModel,
    BertTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)
import torch
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="83945631-f182-436d-a4a2-f07e80aaa3c3")
index = pc.Index("example-namespace-example-index")


def get_pinecone_compatible_vectors(text_chunks):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    max_length = 512
    embeddings = []

    for chunk in text_chunks:
        if isinstance(chunk, list):
            chunk = " ".join(chunk)

        encoding = tokenizer.encode_plus(
            chunk,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            filtered_indices = [
                i
                for i in range(len(input_ids[0]))
                if input_ids[0][i]
                not in [tokenizer.pad_token_id, tokenizer.cls_token_id]
            ]

            filtered_embedding = (
                last_hidden_state[:, filtered_indices, :].mean(dim=1).squeeze()
            )

            embeddings.append(filtered_embedding)

    return embeddings


def query_pinecone(query_text, top_k=5):
    query_vector = get_pinecone_compatible_vectors([query_text])
    query_list = [float(val) for val in query_vector[0]]

    ans = index.query(
        vector=query_list,
        top_k=top_k,
        include_metadata=True,
        include_values=True,
    )
    for match in ans["matches"]:
        text = match["metadata"].get("text", "No text found")
        print(f"ID: {match['id']}, Score: {match['score']}, Text: {text}")
    retrieved_texts = [
        match["metadata"].get("text", "No text found") for match in ans["matches"]
    ]
    return "\n".join(retrieved_texts)


def generate_response(context, question):
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = bart_tokenizer.encode(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )
    outputs = bart_model.generate(
        inputs, max_length=150, num_beams=4, early_stopping=True
    )

    generated_text = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


query_text = "the Wind and the sun"
context = query_pinecone(query_text, top_k=5)
question = "what was the wind and the sun story about?"

generated_response = generate_response(context, question)

print("Generated response:")
print(generated_response)
