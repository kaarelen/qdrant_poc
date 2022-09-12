import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
from gensim.models import Word2Vec

VECTOR_SIZE = 300
collection_name = "alice"
q_client = QdrantClient(host="localhost", port=6333)
q_client.recreate_collection(
    collection_name=collection_name,
    distance=models.Distance.COSINE,
    vector_size=VECTOR_SIZE,
)


def get_prepared_text(file_name: str):
    regex = re.compile("[^a-zA-Z ]")  # remove all non alphabetic characters, exclude spaces
    with open(file_name, "r") as file:
        text = []
        for i in file.read().split("."):
            text.append(regex.sub(" ", i).strip().lower().split())
    return text


def get_model():
    return Word2Vec(
        sentences=get_prepared_text("alice.txt"),
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
    )


def get_points_from_vocabular(vocabular):
    points = []
    for index, word in enumerate(vocabular):
        points.append(
            models.PointStruct(
                id=index,
                payload={
                    "word": word,
                },
                vector=list(model.wv[word]),
            ),
        )
    return points


if __name__ == "__main__":
    target_word = "teacups".lower()
    print(f"target word {target_word}")

    model = get_model()

    print("original (gensim Word2vec):")
    for similar_word in model.wv.most_similar(target_word, topn=30):
        print(similar_word)

    q_client.upsert(
        collection_name=collection_name,
        points=get_points_from_vocabular(model.wv.key_to_index),
    )
    print("similarity from qdrant:")
    for point in q_client.search(
        collection_name=collection_name,
        query_vector=list(model.wv[target_word]),
        limit=30,
    ):
        print(point.payload["word"], point.score)
