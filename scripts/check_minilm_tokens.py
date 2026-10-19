from sentence_transformers import SentenceTransformer
from finrag.documents import load_pdf_pages

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

pages = load_pdf_pages(
    "data/external/financebench/AMD_2022_10K.pdf"
)

print("Límite MiniLM:", model.max_seq_length)

for i in [3, 55]:
    tokens = model.tokenizer.encode(
        pages[i].text,
        truncation=False,
    )
    print(f"Página {i}: {len(tokens)} tokens")

words = pages[55].text.split()

fragment = " ".join(words[:150])

tokens = model.tokenizer.encode(
    fragment,
    truncation=False,
)

print("Palabras:", len(fragment.split()))
print("Tokens:", len(tokens))

from finrag.documents import Document, chunk_document

chunks = []

for page in pages:
    document = Document(
        source=f"{page.doc_name}:p{page.page_index}",
        text=page.text,
    )

    chunks.extend(
        chunk_document(
            document,
            chunk_size=80,
            overlap=15,
        )
    )

lengths = [
    len(model.tokenizer.encode(chunk.text, truncation=False))
    for chunk in chunks
]

print("Chunks generados:", len(chunks))
print("Máximo de tokens:", max(lengths))
print("Chunks que superan 256:", sum(n > 256 for n in lengths))