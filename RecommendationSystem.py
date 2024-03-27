import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the CSV files
df_network = pd.read_csv('AI2Peat_Relationship.csv', delimiter=';')
df_papers = pd.read_csv('AI2Peat_List.csv', sep=';')

# Create the graph
G = nx.from_pandas_edgelist(df_network, source='Paper_Number_Cited', target='Paper_Number_Quoter', create_using=nx.Graph())

# Encode the sentences
sentences = df_papers['Paper_Name'].tolist()
sentence_embeddings = model.encode(sentences)

def is_within_n_nodes(G, start_node, end_node, n=5):
    try:
        shortest_path_length = nx.shortest_path_length(G, source=start_node, target=end_node)
        return shortest_path_length <= n
    except nx.NetworkXNoPath:
        return False

def find_most_similar(input_sentence, sentence_embeddings, sentences, top_k=5, use_graph=True):
    input_embedding = model.encode([input_sentence])
    similarities = cosine_similarity(input_embedding, sentence_embeddings)

    sorted_indices = np.argsort(similarities[0])[::-1]

    filtered_sentences = []
    last_similarity_score = -1  # Initialize with a score that won't be in the results

    if use_graph:
        paper_exists = df_papers['Paper_Name'] == input_sentence
        if not paper_exists.any():
            print("The paper isn't in the database, but here is the recommendation based on your input:")
            use_graph = False  # Skip graph analysis if paper doesn't exist
        else:
            input_paper_number = df_papers[paper_exists]['Paper_Number'].values[0]

    for index in sorted_indices:
        paper_name = sentences[index]
        paper_link = df_papers.iloc[index]['Paper_Link']
        current_similarity_score = similarities[0][index]

        # Skip if the similarity score is the same as the last one or too high, indicating potential self-match
        if current_similarity_score == last_similarity_score or current_similarity_score >= 0.95:
            continue

        if use_graph:
            target_paper_number = df_papers.iloc[index]['Paper_Number']
            if is_within_n_nodes(G, input_paper_number, target_paper_number, n=5):
                filtered_sentences.append((paper_name, current_similarity_score, paper_link))
                if len(filtered_sentences) >= top_k:
                    break  # Stop if we've reached the desired number of results
        else:
            filtered_sentences.append((paper_name, current_similarity_score, paper_link))
            if len(filtered_sentences) >= top_k:
                break  # Also stop for non-graph based approach when top_k is reached

        last_similarity_score = current_similarity_score  # Update last similarity score for next iteration

    return filtered_sentences

input_sentence = input("Enter the name of the paper: ")
similar_sentences = find_most_similar(input_sentence, sentence_embeddings, sentences, top_k=5, use_graph=True)

# Print the recommendations along with their links
for sentence, score, link in similar_sentences:
    print(f"Sentence: {sentence}, Similarity Score: {score}, Link: {link}")
