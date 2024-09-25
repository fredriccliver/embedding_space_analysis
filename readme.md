# Episode Embedding Visualizer and Recommender

This project visualizes episode embeddings in a 3D space using Principal Component Analysis (PCA) and provides episode recommendations based on user interactions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Core Functionality](#core-functionality)
- [Visualization and Results](#visualization-and-results)
- [Embedding to Sentence Model](#embedding-to-sentence-model)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone episode-embedding-visualizer.git
   cd episode-embedding-visualizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have an `episodes.json` file in the project directory containing the episode data with embeddings.

## Usage

To run the visualization and recommender system:

```bash
python index.py
```

This will start a Dash server. Open a web browser and go to http://127.0.0.1:8050/ to view the interactive visualization and recommender system.

## Features

- 3D visualization of episode embeddings using PCA
- Interactive scatter plot with clickable points
- Similar episode recommendations based on cosine similarity
- Cluster-based recommendations using DBSCAN
- Dynamic updating of recommendations based on user interactions
- Ability to clear all favorite episodes with a single button click
- Favorite episodes list
- Color-coded visualization (red for favorites, blue for recommendations)

## Core Functionality

The main functionality is implemented in `index.py`. Here's an overview of what it does:

1. Loads episode data from `episodes.json`.
2. Performs PCA on episode embeddings to reduce dimensionality to 3D.
3. Creates an interactive 3D scatter plot using Plotly and Dash.
4. Provides similar episode recommendations based on cosine similarity.
5. Generates cluster-based recommendations using DBSCAN when 3 or more points are clicked.
6. Dynamically updates the visualization and recommendations based on user interactions.

Key components:
- PCA: Reduces the high-dimensional embeddings to 3D for visualization.
- Cosine Similarity: Calculates similarity between episodes for recommendations.
- DBSCAN: Performs clustering for advanced recommendations.
- Dash and Plotly: Creates an interactive web application with 3D scatter plot.

## Recommendation Logic

The recommendation system in this project uses two main approaches:

### 1. Similarity-Based Recommendations

This method recommends episodes based on their similarity to a given episode or set of episodes.

- **Input**: One or more episode IDs
- **Process**:
  1. Retrieve the embeddings for the input episodes
  2. Calculate the cosine similarity between these embeddings and all other episode embeddings
  3. Rank episodes based on their average similarity score
  4. Select the top N most similar episodes (excluding the input episodes)
- **Output**: List of recommended episode IDs and their similarity scores

### 2. Cluster-Based Recommendations

This method is used when three or more episodes are selected, providing recommendations based on the cluster of the selected episodes.

- **Input**: Three or more episode IDs
- **Process**:
  1. Perform DBSCAN clustering on all episode embeddings
  2. Identify the cluster(s) containing the input episodes
  3. Rank all episodes in the identified cluster(s) based on their average similarity to the input episodes
  4. Select the top N episodes from this ranked list (excluding the input episodes)
- **Output**: List of recommended episode IDs from the same cluster(s)

### Implementation Details

- The similarity calculations use cosine similarity, which measures the cosine of the angle between two vectors in the embedding space.
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is used for clustering, as it can discover clusters of arbitrary shape and doesn't require specifying the number of clusters in advance.
- The system dynamically switches between similarity-based and cluster-based recommendations depending on the number of selected episodes.

## User Interaction

The application provides an interactive user interface with the following features:

1. **3D Scatter Plot**: Users can interact with a 3D scatter plot representing episode embeddings.
2. **Favorite Episodes**: Clicking on a point in the scatter plot adds it to the list of favorite episodes.
3. **Similar Episodes**: For each selected episode, a table of similar episodes with similarity scores is displayed.
4. **Cluster-based Recommendations**: When 3 or more episodes are selected, the system provides cluster-based recommendations.
5. **Clear Favorites**: A button allows users to clear all favorite episodes and reset the visualization.
6. **Color Coding**: Favorite episodes are highlighted in red, while recommended episodes are shown in blue in the scatter plot.

## Data Requirements

The application expects an `episodes.json` file with the following structure for each episode:

- `title`: The title of the episode
- `id`: A unique identifier for the episode
- `embedding`: A string representation of the episode's embedding vector (1536 dimensions)

Ensure this file is present in the project directory before running the application.

## Performance Advice

When integrating this recommendation system into a mobile application, consider the following performance optimizations:

### 1. Pre-computation and Caching

- **Similarity Matrix**: Pre-compute and store the similarity matrix for all episodes. This can significantly reduce computation time for similarity-based recommendations.
- **Clusters**: Pre-compute DBSCAN clusters and store the results. Update these periodically if new episodes are added.
- **Top N Similar**: For each episode, pre-compute and store the top N most similar episodes.

### 2. Efficient Data Storage

- Use efficient data structures for storing embeddings and pre-computed results (e.g., NumPy arrays for embeddings, dictionaries for quick lookups).
- Consider using a lightweight database like SQLite for structured storage and efficient querying.

### 3. Lazy Loading

- Load episode data and embeddings in batches or on-demand, especially if the dataset is large.
- Implement pagination for recommendation results to avoid loading all recommendations at once.

### 4. Asynchronous Processing

- Perform heavy computations (like DBSCAN clustering) in background threads to keep the UI responsive.
- Use asynchronous programming patterns to handle network requests if fetching data from a server.

### 5. Dimensionality Reduction

- If memory is a concern, consider reducing the dimensionality of embeddings using techniques like PCA before storing them on the device.

### 6. Server-Side Computation

- For complex operations or large datasets, consider offloading computations to a server and implementing an API for the mobile app to request recommendations.

### 7. Caching and Persistence

- Cache recommendation results locally to reduce repeated computations.
- Implement a time-based or version-based cache invalidation strategy to ensure recommendations stay up-to-date.

### 8. Optimize for Cold Start

- Implement a fallback recommendation strategy for new users or when there's insufficient interaction data.
- Consider using content-based filtering as an initial recommendation strategy.

### 9. Adaptive Computation

- Adjust the complexity of recommendations based on the device's capabilities and current load.
- Implement simpler recommendation algorithms for low-end devices or when battery is low.

### 10. Profiling and Monitoring

- Use profiling tools to identify performance bottlenecks in your specific implementation.
- Implement analytics to monitor the performance and effectiveness of recommendations in real-world usage.

By applying these optimizations, you can ensure that the recommendation system runs efficiently on mobile devices, providing a smooth user experience while managing computational resources effectively.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Embedding to Sentence Model

The project includes a script `train_embedding_to_sentence_model.py` that trains a model to generate titles from episode embeddings. Here's a brief overview:

### Functionality

- Loads episode data from `episodes.json`
- Defines a custom dataset (`TitleDataset`) for handling embeddings and titles
- Implements a custom `CustomEncoderDecoder` model based on BERT
- Provides functions for training the model and generating titles from embeddings

### Key Components

- Uses `EncoderDecoderModel` from Hugging Face Transformers
- Implements a custom embedding projector to handle 1536-dimensional embeddings
- Includes a `generate_title` function for inference

### Usage

To train the model:

```bash
python train_embedding_to_sentence_model.py
```

The trained model is saved as `embedding_to_sentence_model.pth`. Once trained, you can use the model to generate titles from embeddings.

This model enhances the project by allowing generation of descriptive titles from episode embeddings, which can be useful for content summarization or recommendation explanations.

## Visualization and Results

### Episode Embeddings Visualization and Recommendations

![Episode Embeddings Visualization](screenshot.png)

This screenshot shows the main interface of our application:

- The left side displays a 3D scatter plot of episode embeddings, with each point representing an episode.
- Red points indicate favorite episodes selected by the user.
- Blue points represent recommended episodes based on the user's selections.
- The right side shows:
  - A list of favorite episodes
  - Similar episodes to the last selected favorite, with similarity scores
  - Cluster-based recommendations when 3 or more episodes are selected

This visualization allows users to intuitively explore the relationships between different episodes and receive personalized recommendations.

### Model Training Results

![Loss History](loss_history.png)

This graph shows the training progress of our embedding-to-sentence model:

- The left plot displays the average loss per epoch, showing a steady decrease over time.
- The right plot shows the loss per batch, providing a more detailed view of the training process.
- Both graphs indicate that the model successfully learned to generate titles from embeddings, with the loss converging to a low value.

These results demonstrate the effectiveness of our training process and the model's ability to capture the relationship between episode embeddings and their titles.