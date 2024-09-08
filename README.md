# Movie Recommendation System

This project is a **content-based movie recommendation system** built using Python's **Pandas**, **Scikit-learn**, and **TF-IDF Vectorization**. The system recommends movies similar to the one entered by the user based on the plot descriptions of the movies.

## Features

- **Weighted Rating Calculation**: Uses IMDb’s weighted rating formula to rank movies based on their vote count and average rating.
- **Content-Based Recommendation**: Recommends movies by comparing the textual similarity of plot overviews using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity.
- **Custom Movie Search**: Users can enter any part of a movie title, and the system finds similar movies based on the closest match.

## How It Works

1. The dataset (`movies_metadata.csv`) is loaded and cleaned.
2. A weighted rating system filters and ranks high-quality movies based on their vote count and rating.
3. A **TF-IDF Vectorizer** is applied to the movie's plot overviews to convert text data into numerical features.
4. A **cosine similarity matrix** is computed to find similarities between all movies.
5. The system allows users to input a movie title, and it returns a list of the top 10 most similar movies based on the cosine similarity of the plot descriptions.

## Dataset

The system uses the [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/) from **Kaggle**, which contains information such as movie titles, plot overviews, average ratings, and vote counts.

## Installation

### Prerequisites

- **Python 3.x**
- **Pandas**
- **Scikit-learn**

You can install the required libraries using pip.

### Running the Script
- Download or clone this repository.
- Place your movies_metadata.csv dataset in the same directory as the script.
- Run the script using Python.
- Enter a movie title when prompted. You can search for part of a title, and the system will suggest similar movies based on the plot.
- To exit, type exit.

### Usage
Once the script is running, simply type the name of a movie and press Enter. The system will provide you with a list of similar movies based on the content of their plot overviews.

Example:

Enter a movie name (or type 'exit' to quit): Inception
Movies similar to 'Inception':
- Interstellar
- The Matrix
- The Prestige
- Shutter Island
...
  
### Key Functions
- weighted_rating: This function calculates the weighted rating of movies using IMDb’s formula based on the vote count and vote average.
- get_recom: Takes a movie title as input and returns a list of the top 10 most similar movies based on cosine similarity.
