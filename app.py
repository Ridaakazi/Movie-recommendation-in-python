import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
movies_data = pd.read_csv('archive/movies_metadata.csv', low_memory=False)

# Print first 3 rows
# print(movies_data.head(3))

# Calculate the mean of the vote_average column
cal = movies_data['vote_average'].mean()
# print("Average rating out of 10:", round(cal, 3))

# Calculate the minimum number of votes required to be in the charts
charts = movies_data['vote_count'].quantile(0.90)
# print(charts)

# Filter out all qualified movies to a new DataFrame
good_movies = movies_data.copy().loc[movies_data['vote_count'] >= charts]

# print("original", movies_data.shape)
# print("filtered", good_movies.shape)

# Function for finding the weighted rating of each movie
def weighted_rating(x, charts=charts, cal=cal):
    V = x['vote_count']
    R = x['vote_average']
    # IMDb formula
    return (V / (V + charts) * R) + (charts / (charts + V) * cal)

# Apply the weighted_rating function and create a new 'score' column
good_movies['score'] = good_movies.apply(weighted_rating, axis=1)

# Sort movies based on the 'score' column
good_movies = good_movies.sort_values('score', ascending=False)

# Print the top 15 movies
# print(good_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

# Content-based recommender system

# Print plot overviews of the first 5 movies
# print(movies_data['overview'].head())

# Initialize TF-IDF Vectorizer with English stop words
Tf = TfidfVectorizer(stop_words='english')

# Replace NaN in the 'overview' column with an empty string
movies_data['overview'] = movies_data['overview'].fillna('')

# Construct the TF-IDF matrix(makes it slow)
tf_matrix = Tf.fit_transform(movies_data['overview'])
# print(tf_matrix.shape)

# Display some feature names for verification
# print(Tf.get_feature_names_out()[5000:5010])

print("Scanning Data(5 minutes)")
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tf_matrix, tf_matrix)

# Verify the shape of cosine_sim and the first movie's similarity scores
# print(cosine_sim.shape)
# print(cosine_sim[1])

# Create a Series to map titles to indices
indices = pd.Series(movies_data.index, index=movies_data['title']).drop_duplicates()
# print(indices[:10])
print('Scanning Complete')

# Function that takes a movie title and gives similar movies
def get_recom(title, cosine_sim=cosine_sim):
    # Convert title to lowercase for case-insensitive search
    title_lower = title.lower()
    
    # Find movie titles that contain the input title as a substring (case-insensitive)
    matched_indices = [i for i, t in enumerate(movies_data['title']) if title_lower in str(t).lower()]
    
    if not matched_indices:
        return ["No movies found matching that title."]
    
    idx = matched_indices[0]

    # Get similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (excluding the first one which is the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    sim_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_data['title'].iloc[sim_indices]

# Main loop to keep asking for movie names
while True:
    title = input("Enter a movie name (or type 'exit' to quit): ")
    if title.lower() == 'exit':
        break
    
    recommended_movies = get_recom(title)
    print(f"Movies similar to '{title}':")
    if isinstance(recommended_movies, list) and recommended_movies[0] == "No movies found matching that title.":
        print(recommended_movies[0])
    else:
        for movie in recommended_movies:
            print(movie)

    print()  