# import libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample movie dataset (You can replace this with a real movie dataset)
data = {'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Matrix', 'John Wick', 'The Godfather', 'Pulp Fiction', 'The Dark Knight'],
        'genre': ['Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama']}

# convert the data set into a data frame
df = pd.DataFrame(data)

# display the data set
print("Movie Data: ")
print(df)

# define a tf-idf vectorizer to transform the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

# fit and transform the genre column into a matrix of tf-idf features
tfidf_matrix = tfidf.fit_transform(df['genre'])

# compute the cosine similarity mateix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# recommend movies based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    # get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # get the pairwise similarity of scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort the moves based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the indices of the two most similar movie
    sim_scores = sim_scores[1:3]

    # get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return the titles of the most similar movies
    return df['title'].iloc[movie_indices]


# test the recommendation system with an example
movie_title = 'The Matrix'
recommended_movies = get_recommendations(movie_title)
print(f"Recommended Movie for {movie_title} :")
for movie in recommended_movies:
    print(movie)