import imdb
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from uszipcode import SearchEngine


def main():
    st.title("User Data Clustering")

    # User input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    occupations_list = ["Accountant","Actor","Architect","Artist","Astronaut","Athlete","Author","Baker","Banker","Barista","Bartender","Biologist","Butcher","Carpenter","Chef","Chemist","Civil engineer","Dentist","Doctor","Electrician","Engineer","Firefighter","Flight attendant","Graphic designer","Hairdresser","Journalist","Lawyer","Librarian","Mechanic","Musician","Nurse","Pharmacist","Photographer","Pilot","Police officer","Professor","Programmer","Psychologist","Scientist","Software developer","Teacher","Translator","Veterinarian","Waiter/Waitress","Web developer","Writer"]
    occupation = st.selectbox("Select Occupation", occupations_list)
    occupation = occupations_list.index(occupation)

    df_users = pd.read_csv('users.csv', sep=';')
    unique_zipcodes = df_users['zip-code'].unique()
    selected_zipcode = st.selectbox("Select Zipcode", unique_zipcodes)
    
    # Save user data
    if st.button("Predict Cluster"):
        cluster_number = predict_cluster(gender, age, occupation, selected_zipcode)
        st.success("Predicted Cluster Number: {}".format(cluster_number))

def display_recommendations():
    cluster_number = int(st.experimental_get_query_params().get("cluster", [0])[0])

    # Load your dataset (e.g., movies dataframe)
    df_movies = load_movies_dataset()

    # Filter movies based on cluster number
    clustered_df = df_movies[df_movies['cluster'] == cluster_number]

    # Get the top 5 most common movie titles
    top_movies = clustered_df['title'].value_counts()[:5]

    # Display top movies
    st.subheader("Top Movies in Cluster:")
    for movie_title, count in top_movies.items():
        st.write("- {} (Count: {})".format(movie_title, count))
        # Get movie details from IMDb
        movie_info = get_movie_info(movie_title)
        if movie_info is not None:
            st.image(movie_info['poster'])

    st.markdown("---")  # Add a separator

    # Get the top 5 recommended movies based on ratings
    average_ratings = clustered_df.groupby('title')['rating'].mean()
    sorted_ratings = average_ratings.sort_values(ascending=False)
    recommended_movies = sorted_ratings[:5]

    # Display recommended movies
    st.subheader("Recommended Movies:")
    for movie_title, rating in recommended_movies.items():
        st.write("- {} (Rating: {:.2f})".format(movie_title, rating))
        # Get movie details from IMDb
        movie_info = get_movie_info(movie_title)
        if movie_info is not None:
            st.image(movie_info['poster'])

def content_based_recommendations():
    st.title("Content-Based Recommendations")

    # Load your dataset (e.g., movies dataframe)
    df_movies = load_movies_dataset()

    # Select favorite movie
    favorite_movie = st.selectbox("Select Your Favorite Movie", df_movies['title'].unique())

    # Get top 10 similar movies based on content
    similar_movies = get_similar_movies(favorite_movie, df_movies, top_n=10)

    # Display similar movies
    st.subheader("Top 10 Similar Movies:")
    for index, row in similar_movies.iterrows():
        st.write("- {} (Genres: {})".format(row['title'], row['genres']))
        # Get movie details from IMDb
        movie_info = get_movie_info(row['title'])
        if movie_info is not None:
            st.image(movie_info['poster'])

def item_based_recommendations():
    st.title("Item-Based Recommendations")

    # Load your dataset (e.g., movies dataframe)
    df_movies = pd.read_csv('movies.csv', sep=';', encoding='ISO-8859-1').drop(['Unnamed: 3'], axis=1)
    df_ratings = load_ratings_dataset()

    # Select favorite movie
    favorite_movie = st.selectbox("Select Your Favorite Movie", df_movies['title'].unique())

    # Get top 10 similar movies based on content
    user_item_matrix = df_ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
    movie_similarities = cosine_similarity(user_item_matrix)
    movie_index = df_movies[df_movies['title'] == favorite_movie].index[0]
    similar_movies_indices = movie_similarities[movie_index].argsort()[::-1][1:11]
    similar_movies = df_movies.iloc[similar_movies_indices][['title', 'genres']]
    # Display similar movies
    st.subheader("Top 10 Similar Movies:")
    for index, row in similar_movies.iterrows():
        st.write("- {} (Genres: {})".format(row['title'], row['genres']))
        # Get movie details from IMDb
        movie_info = get_movie_info(row['title'])
        if movie_info is not None:
            st.image(movie_info['poster'])

def predict_cluster(gender, age, occupation, zipcode):
    # Load the KMeans model
    kmeans_model = joblib.load("cluster.h5")

    # Preprocess user data
    user_data = preprocess_data(gender, age, occupation, zipcode)

    # Predict cluster number
    cluster_number = kmeans_model.predict(user_data)
    print(cluster_number[0])

    return cluster_number[0]

# Define a function to get the state for a zip-code
def get_state(zipcode):
    search = SearchEngine()
    result = search.by_zipcode(zipcode)
    if result:
        return result.state
    else:
        return 0

def preprocess_data(gender, age, occupation, zipcode):
    # Perform any necessary preprocessing on the user data
    # For example, convert categorical variables to numerical representations
    gender = 0 if gender == 'Male' else 1

    # Initialize the zip-code search engine
    state = get_state(zipcode)
    state_Encoder = joblib.load("state_Encoder.h5")
    state = state_Encoder.transform([state])

    # Return the preprocessed data as a list or array
    data = [[gender, age, occupation, state[0]]]
    return data

def load_movies_dataset():
    # Load your movies dataset or retrieve it from a database
    # Return the loaded dataset (e.g., dataframe)
    # Replace this with your own code to load the movies dataset
    # Example:
    df_movies = pd.read_csv("merged.csv")
    return df_movies

def load_ratings_dataset():
    # Load your movies dataset or retrieve it from a database
    # Return the loaded dataset (e.g., dataframe)
    # Replace this with your own code to load the movies dataset
    # Example:
    df_ratings = pd.read_csv('ratings.csv', sep=';')
    return df_ratings

def get_similar_movies(selected_movie, df_movies, top_n=10):
    df_movies = pd.read_csv('movies.csv', sep=';', encoding='ISO-8859-1').drop(['Unnamed: 3'], axis=1)
    selected_movie_genres = df_movies.loc[df_movies['title'] == selected_movie, 'genres'].iloc[0].split('|')
    similar_movies = df_movies[df_movies['genres'].apply(lambda x: any(genre in x.split('|') for genre in selected_movie_genres))].copy()
    similar_movies['similarity_score'] = similar_movies['genres'].apply(
        lambda x: len(set(x.split('|')).intersection(selected_movie_genres)) / len(
            set(x.split('|')).union(selected_movie_genres)))
    similar_movies = similar_movies.sort_values('similarity_score', ascending=False)
    top_10_similar_movies = similar_movies.iloc[1:11, :][['title', 'genres']]

    return top_10_similar_movies

def get_movie_info(movie_title):
    # Create an instance of the IMDb class
    ia = imdb.IMDb()

    # Search for the movie title
    movies = ia.search_movie(movie_title)

    if movies:
        movie = movies[0]  # Get the first result
        ia.update(movie)  # Fetch additional information

        # Extract relevant details
        info = {
            'name': movies[0]['title'],
            'poster': movies[0]['cover url']
        }

        return info

    return None

PAGES = {
    "Home": main,
    "Recommendations": display_recommendations,
    "Content-Based Recommendations": content_based_recommendations,
    "Item-Based Recommendations": item_based_recommendations
}

def run_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))

    PAGES[page]()

if __name__ == "__main__":
    run_app()
