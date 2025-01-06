import matplotlib.pyplot as plt
import networkx as nx

nodes = [
    "Program 1:\nTraining\nModule",
    "Load\nData",
    "Preprocess\nText",
    "Extract\nFeatures",
    "Train\nModel",
    "Evaluate\nModel",
    "Save\nArtifacts",
    "Model",
    "Vectorizer",
    "Program 2:\nPrediction\nModule",
    "Load\nArtifacts",
    "Input\nMessage",
    "Make\nPrediction",
    "Output\nResult",
]

edges = [
    ("Program 1:\nTraining\nModule", "Load\nData"),
    ("Load\nData", "Preprocess\nText"),
    ("Preprocess\nText", "Extract\nFeatures"),
    ("Extract\nFeatures", "Train\nModel"),
    ("Train\nModel", "Evaluate\nModel"),
    ("Train\nModel", "Save\nArtifacts"),
    ("Save\nArtifacts", "Model"),
    ("Save\nArtifacts", "Vectorizer"),
    ("Program 2:\nPrediction\nModule", "Load\nArtifacts"),
    ("Model", "Load\nArtifacts"),
    ("Vectorizer", "Load\nArtifacts"),
    ("Load\nArtifacts", "Input\nMessage"),
    ("Input\nMessage", "Preprocess\nText"),
    ("Preprocess\nText", "Extract\nFeatures"),
    ("Extract\nFeatures", "Make\nPrediction"),
    ("Make\nPrediction", "Output\nResult"),
]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

pos_hierarchy = {
    "Program 1:\nTraining\nModule": (0, 4),
    "Load\nData": (1, 3.5),
    "Preprocess\nText": (2, 3),
    "Extract\nFeatures": (3, 2.5),
    "Train\nModel": (4, 2),
    "Evaluate\nModel": (5, 2.5),
    "Save\nArtifacts": (5, 1.5),
    "Model": (4, 1),
    "Vectorizer": (5, 0.5),
    "Program 2:\nPrediction\nModule": (0, 0),
    "Load\nArtifacts": (1, -0.5),
    "Input\nMessage": (2, -1),
    "Make\nPrediction": (3, -1.5),
    "Output\nResult": (4, -2),
}

node_colors = []
for node in nodes:
    if "Program" in node:  
        node_colors.append("lightgreen")
    elif node == "Model" or node == "Vectorizer":
        node_colors.append("gold")
    else: 
        node_colors.append("skyblue")

node_sizes = []
for node in nodes:
    if "Program" in node:  
        node_sizes.append(7000)
    else:
        node_sizes.append(4000)

plt.figure(figsize=(14, 10))

nx.draw(
    G,
    pos=pos_hierarchy,
    with_labels=True,
    node_size=node_sizes,
    node_color=node_colors,
    font_size=10,
    font_weight="bold",
    arrowsize=40,
    edge_color="darkgray",
    font_color="black",
)

plt.axis("off")
plt.show()