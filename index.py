import dash
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table
from scipy.spatial import ConvexHull
from dash.dependencies import Input, Output, State, ALL
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from sklearn.cluster import KMeans
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data from episodes.json
with open('episodes.json', 'r', encoding='utf-8') as f:
    episodes_data = json.load(f)

# Extract titles, summary embeddings, and ids
titles = [episode['title'] for episode in episodes_data]
ids = [episode['id'] for episode in episodes_data]  # Add this line

# Convert string representations of lists to actual lists of floats
embeddings = []
for episode in episodes_data:
    embedding_string = episode['embedding']
    embedding_list = embedding_string.strip('[]').split(',')
    embedding_floats = [float(x.strip()) for x in embedding_list]
    embeddings.append(embedding_floats)

# Convert summary embeddings to numpy array
embeddings_array = np.array(embeddings)

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings_array)

# Create a DataFrame with PCA results and ids
df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
df['title'] = titles
df['id'] = ids  # Add this line

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings_array)

# Create the Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Function to create initial 3D scatter plot


def create_initial_scatter():
    scatter = go.Figure(data=[go.Scatter3d(
        x=df['PC1'],
        y=df['PC2'],
        z=df['PC3'],
        mode='markers',
        text=df['title'],
        customdata=df['id'],  # Use 'id' instead of 'title'
        hoverinfo='text',
        marker=dict(
            size=5,
            color=df['PC1'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    scatter.update_layout(
        title='Episode Embeddings Visualization',
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        ),
        width=800,
        height=900,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    return scatter


# Define the layout
app.layout = html.Div([
    html.H1("Episode Embeddings Visualization and Recommendations",
            style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot', figure=create_initial_scatter(),
                      style={'height': '80vh'}),
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Favorite Episodes:", style={'marginTop': '20px'}),
            html.Button('Clear Favorites', id='clear-favorites-button', n_clicks=0),  # New Clear button
            html.Div(id='favorite-episodes', style={'marginBottom': '20px'}),
            html.H3("Similar Episodes:"),
            html.Div(id='table-container', style={'marginBottom': '20px'}),
            html.H3("Cluster-based Recommendations:"),
            html.Div(id='cluster-recommendations', style={'border': '1px solid black', 'padding': '10px', 'minHeight': '100px'})
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '20px', 'overflowY': 'auto', 'height': '80vh'})
    ]),
    # dcc.Store(id='highlight-store', data={'current_highlight': None}),
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1400px'})

# Global variable to store favorite points
favorite_points = []


def generate_cluster_name(cluster_indices, titles):
    # Get the titles and descriptions of episodes in this cluster
    cluster_texts = [f"{titles[i]}" for i in cluster_indices]

    # Use TF-IDF to find important words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)

    # Get the top 5 words with highest TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_sums = tfidf_matrix.sum(axis=0).A1
    top_words = [feature_names[i] for i in tfidf_sums.argsort()[-5:][::-1]] 

    return " ".join(top_words).title()

# Function to create convex hull


def create_convex_hull(points):
    if len(points) < 3:
        return points  # Not enough points for a convex hull

    hull = ConvexHull(points)
    return points[hull.vertices]


def optimal_eps(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    return np.mean(distances)


def clear_favorites(figure):
    # Reset marker sizes to original (assuming 5 is the original size)
    figure['data'][0]['marker']['size'] = 5

    # Reset colors to original (based on PC1)
    figure['data'][0]['marker']['color'] = df['PC1']
    figure['data'][0]['marker']['colorscale'] = 'Viridis'
    
    # Clear the favorite episodes list
    favorite_list = []
    
    # Clear the recommended episodes list
    recommended_list = []
    
    return figure, favorite_list, recommended_list


@app.callback(
    [Output('table-container', 'children'),
     Output('favorite-episodes', 'children'),
     Output('cluster-recommendations', 'children'),
     Output('scatter-plot', 'figure')],
    [Input('scatter-plot', 'clickData'),
     Input('clear-favorites-button', 'n_clicks')],
    [State('scatter-plot', 'figure'),
     State('favorite-episodes', 'children')]
)
def update_table_and_recommendations(clickData, clear_clicks, current_figure, favorite_list):
    global favorite_points
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'clear-favorites-button':
        # Clear all favorite episodes
        cleared_figure, cleared_favorites, cleared_recommendations = clear_favorites(current_figure)
        favorite_points = []  # Reset the global favorite_points list
        return [], cleared_favorites, cleared_recommendations, cleared_figure
    
    if clickData is None:
        return "Click a point to see similar episodes", "No favorite episodes yet", "", current_figure

    point = clickData['points'][0]
    favorite_id = point['customdata']
    favorite_title = point['text']

    if favorite_id not in favorite_points:
        favorite_points.append(favorite_id)

    print(f"Updated favorite points: {favorite_points}")

    # Update similar episodes table
    similarities = similarity_matrix[ids.index(favorite_id)]
    sorted_indices = np.argsort(similarities)[::-1]
    top_n = 5
    similar_titles = [titles[i] for i in sorted_indices[1:top_n+1]]
    similarity_scores = [f"{similarities[i]*100:.2f}%" for i in sorted_indices[1:top_n+1]]

    table = html.Div([
        html.H4(f"Last favorite: {favorite_title}", style={'fontStyle': 'italic', 'color': '#555'}),
        dash_table.DataTable(
            columns=[
                {"name": "Title", "id": "title"},
                {"name": "Similarity Score", "id": "score"}
            ],
            data=[
                {"title": title, "score": score}
                for title, score in zip(similar_titles, similarity_scores)
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])

    # Update favorite points list
    favorite_points_text = html.Ul([html.Li(df[df['id'] == point_id]['title'].iloc[0]) for point_id in favorite_points])

    # Generate cluster recommendations
    if len(favorite_points) >= 3:
        print("Generating cluster recommendations...")
        favorite_indices = [ids.index(id) for id in favorite_points]
        favorite_embeddings = embeddings_array[favorite_indices]

        print(f"Favorite indices: {favorite_indices}")
        print(f"Shape of favorite embeddings: {favorite_embeddings.shape}")

        # Find nearest neighbors of favorite points
        n_neighbors = min(100, len(embeddings_array))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(embeddings_array)
        distances, indices = nn.kneighbors(favorite_embeddings)

        print(f"Shape of nearest neighbors indices: {indices.shape}")

        # Flatten and unique-ify the indices
        neighborhood_indices = np.unique(indices.flatten())
        print(f"Number of unique neighborhood indices: {len(neighborhood_indices)}")

        # Use the original PCA results for the neighborhood
        neighborhood_pca = pca_result[neighborhood_indices]
        print(f"Shape of neighborhood PCA: {neighborhood_pca.shape}")

        # Perform DBSCAN clustering on the neighborhood PCA
        eps = optimal_eps(neighborhood_pca, 5)
        print(f"Calculated eps for DBSCAN: {eps}")
        dbscan = DBSCAN(eps=eps, min_samples=3)
        cluster_labels = dbscan.fit_predict(neighborhood_pca)

        print(f"Unique cluster labels: {np.unique(cluster_labels)}")
        print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")

        # Find the cluster centers
        unique_labels = set(cluster_labels)
        cluster_centers = []
        cluster_names = {}
        for label in unique_labels:
            if label != -1:  # -1 is the label for noise points
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_names[label] = generate_cluster_name(cluster_indices, [titles[i] for i in neighborhood_indices])
                cluster_center = neighborhood_pca[cluster_labels == label].mean(axis=0)
                cluster_centers.append(cluster_center)

        print(f"Cluster names: {cluster_names}")

        # Find episodes closest to the cluster centers
        recommendations = []
        for label, center in zip(unique_labels, cluster_centers):
            if label != -1:
                distances = np.linalg.norm(neighborhood_pca - center, axis=1)
                closest_indices = np.argsort(distances)[:5]
                recommendations.extend([(ids[neighborhood_indices[i]], cluster_names[label])
                                       for i in closest_indices if ids[neighborhood_indices[i]] not in favorite_points])

        print(f"Number of initial recommendations: {len(recommendations)}")

        # Increase the number of initial recommendations
        recommendations = list(dict.fromkeys(recommendations))[:20]
        print(f"Number of unique recommendations: {len(recommendations)}")

        # Calculate overall recommendation scores
        recommendation_scores = []
        for rec_id, cluster in recommendations:
            rec_index = ids.index(rec_id)
            max_similarity = max([similarity_matrix[rec_index][ids.index(favorite_id)] for favorite_id in favorite_points])
            recommendation_scores.append(max_similarity)

        # Sort recommendations by score and take top 10
        sorted_recommendations = sorted(zip(recommendations, recommendation_scores), key=lambda x: x[1], reverse=True)[:10]

        print(f"Generated {len(sorted_recommendations)} final recommendations")
        if len(sorted_recommendations) > 0:
            cluster_recommendations = html.Ul([
                html.Li([
                    html.Span(f"{score*100:.2f}%"),
                    html.Br(),
                    html.Span(df[df['id'] == rec_id]['title'].iloc[0]),
                    html.Br(),
                    html.Span(f"(Cluster Rep. words: {cluster})")
                ],
                    id={'type': 'recommendation', 'index': i},
                    n_clicks=0,
                    style={'cursor': 'pointer'}
                )
                for i, ((rec_id, cluster), score) in enumerate(sorted_recommendations)
            ])
        else:
            cluster_recommendations = html.Div("No recommendations generated. Try selecting more diverse points.")

        # Update the scatter plot colors
        colors = ['grey'] * len(df)  # Initialize all colors to grey

        # Set favorite points to red
        for favorite_id in favorite_points:
            index = df[df['id'] == favorite_id].index[0]
            colors[index] = 'red'

        recommended_ids = [rec_id for (rec_id, _), _ in sorted_recommendations]
        for rec_id in recommended_ids:
            index = df[df['id'] == rec_id].index[0]
            if colors[index] != 'red':  # Only change to blue if it's not already a favorite (red)
                colors[index] = 'blue'

        current_figure['data'][0]['marker']['color'] = colors

    else:
        print("Not enough points for cluster recommendations")
        cluster_recommendations = html.Div(
            "Click at least 3 points to see cluster-based recommendations")

    print("Returning updated data")
    return table, favorite_points_text, cluster_recommendations, current_figure


if __name__ == '__main__':
    app.run_server(debug=True)

print("Dash app is running. Please open a web browser and go to http://127.0.0.1:8050/ to view the interactive plot.")
