import imdb
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uszipcode import SearchEngine
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache functions for performance
@st.cache_data(ttl=3600)
def load_movies_dataset():
    """Load and cache the movies dataset"""
    try:
        df_movies = pd.read_csv("merged.csv")
        return df_movies
    except Exception as e:
        logger.error(f"Error loading movies dataset: {e}")
        st.error("Failed to load movies dataset. Please check if the file exists.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ratings_dataset():
    """Load and cache the ratings dataset"""
    try:
        df_ratings = pd.read_csv('ratings.csv', sep=';')
        return df_ratings
    except Exception as e:
        logger.error(f"Error loading ratings dataset: {e}")
        st.error("Failed to load ratings dataset. Please check if the file exists.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_users_dataset():
    """Load and cache the users dataset"""
    try:
        df_users = pd.read_csv('users.csv', sep=';')
        return df_users
    except Exception as e:
        logger.error(f"Error loading users dataset: {e}")
        st.error("Failed to load users dataset. Please check if the file exists.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_movies_csv():
    """Load and cache the movies.csv dataset"""
    try:
        df_movies = pd.read_csv('movies.csv', sep=';', encoding='ISO-8859-1')
        if 'Unnamed: 3' in df_movies.columns:
            df_movies = df_movies.drop(['Unnamed: 3'], axis=1)
        return df_movies
    except Exception as e:
        logger.error(f"Error loading movies.csv dataset: {e}")
        st.error("Failed to load movies.csv dataset. Please check if the file exists.")
        return pd.DataFrame()

@st.cache_resource
def load_kmeans_model():
    """Load the KMeans clustering model"""
    try:
        return joblib.load("cluster.h5")
    except Exception as e:
        logger.error(f"Error loading KMeans model: {e}")
        st.error("Failed to load clustering model. Please check if the file exists.")
        return None

@st.cache_resource
def load_state_encoder():
    """Load the state encoder"""
    try:
        return joblib.load("state_Encoder.h5")
    except Exception as e:
        logger.error(f"Error loading state encoder: {e}")
        st.error("Failed to load state encoder. Please check if the file exists.")
        return None

@st.cache_data(ttl=86400)
def get_state(zipcode):
    """Get the state for a given zip code"""
    try:
        search = SearchEngine()
        result = search.by_zipcode(zipcode)
        if result:
            return result.state
        else:
            return "Unknown"
    except Exception as e:
        logger.error(f"Error getting state for zipcode {zipcode}: {e}")
        return "Unknown"

@st.cache_data(ttl=86400)
def get_movie_info(movie_title):
    """Get movie information from IMDb"""
    try:
        # Create an instance of the IMDb class
        ia = imdb.IMDb()

        # Search for the movie title
        movies = ia.search_movie(movie_title)

        if movies:
            movie = movies[0]  # Get the first result
            ia.update(movie)  # Fetch additional information

            # Extract relevant details
            info = {
                'name': movie.get('title', 'Unknown title'),
                'poster': movie.get('full-size cover url', movie.get('cover url', '')),
                'year': movie.get('year', 'Unknown year'),
                'director': ', '.join([d.get('name', 'Unknown') for d in movie.get('directors', [])]) if 'directors' in movie else 'Unknown',
                'cast': ', '.join([a.get('name', 'Unknown') for a in movie.get('cast', [])[:5]]) if 'cast' in movie else 'Unknown',
                'rating': movie.get('rating', 'Not rated'),
                'plot': movie.get('plot outline', 'No plot available')
            }

            return info
        return None
    except Exception as e:
        logger.error(f"Error getting movie info for {movie_title}: {e}")
        return None

def display_movie_card(movie_title, rating=None, count=None, genres=None, movie_info=None):
    """Display a movie card with details and poster"""
    if movie_info is None:
        movie_info = get_movie_info(movie_title)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if movie_info and movie_info['poster']:
            try:
                response = requests.get(movie_info['poster'])
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=150)
                else:
                    st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
            except Exception:
                st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
        else:
            st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)

    with col2:
        st.subheader(movie_title)
        if movie_info:
            st.write(f"**Year:** {movie_info['year']}")
            st.write(f"**Director:** {movie_info['director']}")
            st.write(f"**Cast:** {movie_info['cast']}")
            st.write(f"**IMDb Rating:** {movie_info['rating']}")
            
            if rating is not None:
                st.write(f"**User Rating:** {rating:.2f}/5.0")
            
            if count is not None:
                st.write(f"**Viewed by:** {count} users in your cluster")
                
            if genres:
                st.write(f"**Genres:** {genres}")
                
            with st.expander("Plot Summary"):
                st.write(movie_info['plot'])
        else:
            st.write("No additional information available")

        if rating is not None or count is not None:
            if st.button(f"More like '{movie_title}'", key=f"more_{movie_title}"):
                st.session_state['favorite_movie'] = movie_title
                st.session_state['current_page'] = "Content-Based Recommendations"
                st.rerun()

def preprocess_data(gender, age, occupation, zipcode):
    """Preprocess user data for prediction"""
    try:
        # Convert gender to binary representation
        gender_numeric = 0 if gender == 'Male' else 1

        # Get state from zipcode
        state = get_state(zipcode)
        
        # Encode the state
        state_encoder = load_state_encoder()
        if state_encoder and state != "Unknown":
            state_encoded = state_encoder.transform([state])[0]
        else:
            # Use a default value if encoder fails or state is unknown
            state_encoded = 0
            
        # Return the preprocessed data as a list or array
        data = [[gender_numeric, age, occupation, state_encoded]]
        return data
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        st.error(f"Error preprocessing user data: {e}")
        return [[0, 30, 0, 0]]  # Default fallback values

def predict_cluster(gender, age, occupation, zipcode):
    """Predict the cluster for a user"""
    try:
        # Load the KMeans model
        kmeans_model = load_kmeans_model()
        if kmeans_model is None:
            return 0

        # Preprocess user data
        user_data = preprocess_data(gender, age, occupation, zipcode)

        # Predict cluster number
        cluster_number = kmeans_model.predict(user_data)[0]
        return cluster_number
    except Exception as e:
        logger.error(f"Error predicting cluster: {e}")
        st.error(f"Error predicting user cluster: {e}")
        return 0

def get_similar_movies(selected_movie, top_n=10):
    """Get similar movies based on genre similarity"""
    try:
        df_movies = load_movies_csv()
        
        if df_movies.empty or selected_movie not in df_movies['title'].values:
            return pd.DataFrame()
            
        selected_movie_genres = df_movies.loc[df_movies['title'] == selected_movie, 'genres'].iloc[0].split('|')
        
        # Create a function to calculate Jaccard similarity
        def calculate_jaccard_similarity(genres):
            genres_set = set(genres.split('|'))
            selected_set = set(selected_movie_genres)
            intersection = len(genres_set.intersection(selected_set))
            union = len(genres_set.union(selected_set))
            return intersection / union if union > 0 else 0
        
        # Apply similarity calculation
        similar_movies = df_movies.copy()
        similar_movies['similarity_score'] = similar_movies['genres'].apply(calculate_jaccard_similarity)
        
        # Filter out the selected movie and sort by similarity
        similar_movies = similar_movies[similar_movies['title'] != selected_movie]
        similar_movies = similar_movies.sort_values('similarity_score', ascending=False)
        
        # Get top N similar movies
        top_similar_movies = similar_movies.head(top_n)[['title', 'genres', 'similarity_score']]
        
        return top_similar_movies
    except Exception as e:
        logger.error(f"Error finding similar movies: {e}")
        st.error(f"Error finding similar movies: {e}")
        return pd.DataFrame()

def compute_item_based_recommendations(favorite_movie, top_n=10):
    """Compute item-based collaborative filtering recommendations"""
    try:
        df_movies = load_movies_csv()
        df_ratings = load_ratings_dataset()
        
        if df_movies.empty or df_ratings.empty or favorite_movie not in df_movies['title'].values:
            return pd.DataFrame()
        
        # Create user-item matrix
        user_item_matrix = df_ratings.pivot_table(
            index='movieId', 
            columns='userId', 
            values='rating'
        ).fillna(0)
        
        # Compute similarity matrix
        movie_similarities = cosine_similarity(user_item_matrix)
        
        # Get the index of the favorite movie
        movie_id = df_movies[df_movies['title'] == favorite_movie]['movieId'].iloc[0]
        movie_index = user_item_matrix.index.get_loc(movie_id) if movie_id in user_item_matrix.index else -1
        
        if movie_index == -1:
            return pd.DataFrame()
        
        # Get similar movie indices
        similar_indices = movie_similarities[movie_index].argsort()[::-1][1:top_n+1]
        similar_movie_ids = user_item_matrix.index[similar_indices]
        
        # Get similar movies details
        similar_movies = df_movies[df_movies['movieId'].isin(similar_movie_ids)][['title', 'genres']]
        
        # Add similarity scores
        similar_movies['similarity_score'] = [movie_similarities[movie_index][idx] for idx in similar_indices]
        similar_movies = similar_movies.sort_values('similarity_score', ascending=False)
        
        return similar_movies
    except Exception as e:
        logger.error(f"Error computing item-based recommendations: {e}")
        st.error(f"Error computing item-based recommendations: {e}")
        return pd.DataFrame()

def get_cluster_statistics(cluster_number):
    """Get statistics about a cluster"""
    try:
        df_movies = load_movies_dataset()
        
        if df_movies.empty:
            return None
            
        # Filter movies for the selected cluster
        cluster_movies = df_movies[df_movies['cluster'] == cluster_number]
        
        if cluster_movies.empty:
            return None
            
        # Calculate statistics
        stats = {
            "user_count": len(cluster_movies['userId'].unique()),
            "movie_count": len(cluster_movies['title'].unique()),
            "rating_avg": cluster_movies['rating'].mean(),
            "top_genres": []
        }
        
        # Calculate top genres
        if 'genres' in cluster_movies.columns:
            all_genres = []
            for genres in cluster_movies['genres'].str.split('|'):
                if isinstance(genres, list):
                    all_genres.extend(genres)
            
            genre_counts = pd.Series(all_genres).value_counts().head(5)
            stats["top_genres"] = [(genre, count) for genre, count in genre_counts.items()]
        
        return stats
    except Exception as e:
        logger.error(f"Error getting cluster statistics: {e}")
        return None

def home_page():
    """Home page with user input and cluster prediction"""
    st.title("🎬 Movie Recommendation System")
    st.write("Welcome to the Movie Recommendation System! Enter your details to get personalized movie recommendations.")
    
    with st.container():
        st.subheader("Your Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", min_value=18, max_value=100, value=30)
            
        with col2:
            # Occupations with proper capitalization and spaces
            occupations_list = [
                "Accountant", "Actor", "Architect", "Artist", "Astronaut", "Athlete", 
                "Author", "Baker", "Banker", "Barista", "Bartender", "Biologist", 
                "Butcher", "Carpenter", "Chef", "Chemist", "Civil Engineer", "Dentist", 
                "Doctor", "Electrician", "Engineer", "Firefighter", "Flight Attendant", 
                "Graphic Designer", "Hairdresser", "Journalist", "Lawyer", "Librarian", 
                "Mechanic", "Musician", "Nurse", "Pharmacist", "Photographer", "Pilot", 
                "Police Officer", "Professor", "Programmer", "Psychologist", "Scientist", 
                "Software Developer", "Teacher", "Translator", "Veterinarian", 
                "Waiter/Waitress", "Web Developer", "Writer"
            ]
            
            # Get display occupation and index
            occupation_display = st.selectbox("Occupation", occupations_list)
            # Convert to index for the model (assuming the original code used indices)
            occupation_index = occupations_list.index(occupation_display)
            
            # Load users dataset for zipcodes
            df_users = load_users_dataset()
            
            if not df_users.empty and 'zip-code' in df_users.columns:
                unique_zipcodes = sorted(df_users['zip-code'].unique())
                selected_zipcode = st.selectbox("Zip Code", unique_zipcodes)
            else:
                selected_zipcode = st.text_input("Zip Code", "90210")
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("Click 'Find My Recommendations' to discover movies that match your profile!")
            
        with col2:
            if st.button("Find My Recommendations", type="primary"):
                with st.spinner("Analyzing your profile..."):
                    # Add a small delay for effect
                    time.sleep(0.5)
                    
                    # Predict the cluster
                    cluster_number = predict_cluster(gender, age, occupation_index, selected_zipcode)
                    
                    # Store in session state
                    st.session_state['user_cluster'] = cluster_number
                    st.session_state['current_page'] = "Recommendations"
                    
                    # Notify and redirect
                    st.success(f"Profile analyzed! You belong to cluster {cluster_number}")
                    time.sleep(1)
                    st.rerun()
    
    # About section at the bottom
    with st.expander("About this App"):
        st.write("""
        This Movie Recommendation System uses machine learning to cluster users based on their demographic 
        information and provide personalized movie recommendations. The system uses:
        
        - **Collaborative Filtering**: Recommends movies that similar users have enjoyed
        - **Content-Based Filtering**: Recommends movies that are similar to movies you like
        - **Item-Based Recommendations**: Recommends movies based on movie-to-movie similarity
        
        The recommendations improve as more users interact with the system. Enjoy discovering new movies!
        """)

def recommendations_page():
    """Page showing recommendations based on user cluster"""
    st.title("🍿 Your Personalized Recommendations")
    
    # Get the cluster number from session state
    if 'user_cluster' not in st.session_state:
        st.warning("Please complete your profile on the Home page first.")
        if st.button("Go to Home page"):
            st.session_state['current_page'] = "Home"
            st.rerun()
        return
    
    cluster_number = st.session_state['user_cluster']
    
    # Load dataset
    df_movies = load_movies_dataset()
    
    if df_movies.empty:
        st.error("Could not load movie data. Please try again later.")
        return
    
    # Filter movies for the selected cluster
    clustered_df = df_movies[df_movies['cluster'] == cluster_number]
    
    if clustered_df.empty:
        st.warning(f"No data available for cluster {cluster_number}. Try a different profile.")
        return
    
    # Get cluster statistics
    cluster_stats = get_cluster_statistics(cluster_number)
    
    if cluster_stats:
        # Display cluster insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Users in Cluster", value=cluster_stats["user_count"])
        
        with col2:
            st.metric(label="Unique Movies", value=cluster_stats["movie_count"])
        
        with col3:
            st.metric(label="Average Rating", value=f"{cluster_stats['rating_avg']:.1f}/5.0")
        
        # Show top genres if available
        if cluster_stats["top_genres"]:
            st.subheader("Top Genres in Your Cluster")
            
            # Create data for a bar chart
            genres = [genre for genre, _ in cluster_stats["top_genres"]]
            counts = [count for _, count in cluster_stats["top_genres"]]
            
            # Create a bar chart
            fig = px.bar(
                x=genres, 
                y=counts,
                labels={'x': 'Genre', 'y': 'Count'},
                color=counts,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                title="Popular Genres",
                xaxis_tickangle=-45,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Get top movies by popularity (count) in the cluster
    top_movies_by_count = clustered_df['title'].value_counts().head(5)
    
    if not top_movies_by_count.empty:
        st.header("Most Popular Movies in Your Cluster")
        st.write("These movies are frequently watched by people similar to you:")
        
        for movie_title, count in top_movies_by_count.items():
            with st.container():
                display_movie_card(movie_title, count=count)
                st.markdown("---")
    
    # Get top movies by rating in the cluster
    st.header("Highly Rated Movies in Your Cluster")
    st.write("These movies are highly rated by people similar to you:")
    
    # Calculate average ratings
    average_ratings = clustered_df.groupby('title')['rating'].agg(['mean', 'count'])
    # Filter to only include movies with at least 3 ratings
    average_ratings = average_ratings[average_ratings['count'] >= 3]
    # Sort by average rating
    sorted_ratings = average_ratings.sort_values(by='mean', ascending=False)
    # Get top 5 movies
    top_rated_movies = sorted_ratings.head(5)
    
    if not top_rated_movies.empty:
        for movie_title, (avg_rating, count) in zip(top_rated_movies.index, top_rated_movies.values):
            with st.container():
                display_movie_card(movie_title, rating=avg_rating, count=count)
                st.markdown("---")
    else:
        st.info("Not enough rating data available for this cluster. Try the content-based recommendations!")

def content_based_recommendations_page():
    """Page showing content-based recommendations"""
    st.title("🎯 Content-Based Recommendations")
    st.write("Get recommendations based on movies with similar genres and themes.")
    
    # Load movies dataset
    df_movies = load_movies_csv()
    
    if df_movies.empty:
        st.error("Could not load movie data. Please try again later.")
        return
    
    # Select favorite movie
    if 'favorite_movie' in st.session_state:
        default_movie = st.session_state['favorite_movie']
        # Clear it after use
        del st.session_state['favorite_movie']
    else:
        default_movie = None
    
    # Prepare options for the search box
    movie_options = sorted(df_movies['title'].unique())
    
    # Use st.selectbox with autocomplete
    favorite_movie = st.selectbox(
        "Select a movie you like:",
        options=movie_options,
        index=movie_options.index(default_movie) if default_movie in movie_options else 0
    )
    
    if favorite_movie:
        # Display the selected movie first
        st.subheader("Selected Movie")
        selected_movie_genres = df_movies.loc[df_movies['title'] == favorite_movie, 'genres'].iloc[0]
        display_movie_card(favorite_movie, genres=selected_movie_genres)
        st.markdown("---")
        
        # Get similar movies
        st.subheader("Similar Movies You Might Enjoy")
        
        with st.spinner("Finding similar movies..."):
            similar_movies = get_similar_movies(favorite_movie, top_n=6)
        
        if not similar_movies.empty:
            for _, row in similar_movies.iterrows():
                with st.container():
                    display_movie_card(row['title'], genres=row['genres'])
                    st.markdown("---")
        else:
            st.info("No similar movies found. Try selecting a different movie.")
    else:
        st.info("Please select a movie to get recommendations.")

def item_based_recommendations_page():
    """Page showing item-based collaborative filtering recommendations"""
    st.title("👥 Collaborative Filtering Recommendations")
    st.write("Get recommendations based on what other users with similar tastes have enjoyed.")
    
    # Load datasets
    df_movies = load_movies_csv()
    df_ratings = load_ratings_dataset()
    
    if df_movies.empty or df_ratings.empty:
        st.error("Could not load required data. Please try again later.")
        return
    
    # Select favorite movie (with default if coming from another page)
    if 'favorite_movie' in st.session_state:
        default_movie = st.session_state['favorite_movie']
        # Clear it after use
        del st.session_state['favorite_movie']
    else:
        default_movie = None
    
    # Prepare options
    movie_options = sorted(df_movies['title'].unique())
    
    # Use st.selectbox with autocomplete
    favorite_movie = st.selectbox(
        "Select a movie you like:",
        options=movie_options,
        index=movie_options.index(default_movie) if default_movie in movie_options else 0
    )
    
    if favorite_movie:
        # Display the selected movie
        st.subheader("Selected Movie")
        selected_movie_genres = df_movies.loc[df_movies['title'] == favorite_movie, 'genres'].iloc[0]
        display_movie_card(favorite_movie, genres=selected_movie_genres)
        st.markdown("---")
        
        # Get item-based recommendations
        st.subheader("Users Who Liked This Also Enjoyed")
        
        with st.spinner("Finding recommendations based on user ratings..."):
            similar_movies = compute_item_based_recommendations(favorite_movie, top_n=6)
        
        if not similar_movies.empty:
            for _, row in similar_movies.iterrows():
                with st.container():
                    # Convert similarity score to a percentage for display
                    similarity = row['similarity_score'] * 100
                    st.write(f"**Similarity:** {similarity:.1f}%")
                    display_movie_card(row['title'], genres=row['genres'])
                    st.markdown("---")
        else:
            st.info("No collaborative recommendations found. Try the content-based recommendations!")
    else:
        st.info("Please select a movie to get recommendations.")

def compare_recommendations_page():
    """Page to compare different recommendation methods"""
    st.title("🔍 Compare Recommendation Methods")
    st.write("See how different recommendation algorithms suggest movies for you.")
    
    # Load datasets
    df_movies = load_movies_csv()
    
    if df_movies.empty:
        st.error("Could not load movie data. Please try again later.")
        return
    
    # Get movie selection
    movie_options = sorted(df_movies['title'].unique())
    favorite_movie = st.selectbox("Select a movie you like:", options=movie_options)
    
    if favorite_movie:
        tabs = st.tabs(["Content-Based", "Collaborative Filtering", "Comparison"])
        
        with tabs[0]:
            st.subheader("Content-Based Recommendations")
            st.write("Based on movie genres and features")
            
            with st.spinner("Finding similar movies..."):
                content_recs = get_similar_movies(favorite_movie, top_n=5)
            
            if not content_recs.empty:
                for _, row in content_recs.iterrows():
                    with st.container():
                        st.write(f"**Similarity Score:** {row['similarity_score']:.2f}")
                        display_movie_card(row['title'], genres=row['genres'])
                        st.markdown("---")
            else:
                st.info("No content-based recommendations available.")
        
        with tabs[1]:
            st.subheader("Collaborative Filtering Recommendations")
            st.write("Based on what other users with similar tastes enjoyed")
            
            with st.spinner("Finding recommendations based on user ratings..."):
                collab_recs = compute_item_based_recommendations(favorite_movie, top_n=5)
            
            if not collab_recs.empty:
                for _, row in collab_recs.iterrows():
                    with st.container():
                        st.write(f"**Similarity Score:** {row['similarity_score']:.2f}")
                        display_movie_card(row['title'], genres=row['genres'])
                        st.markdown("---")
            else:
                st.info("No collaborative filtering recommendations available.")
        
        with tabs[2]:
            st.subheader("Recommendation Methods Comparison")
            
            # Combine and compare recommendations
            content_titles = set(content_recs['title'].tolist()) if not content_recs.empty else set()
            collab_titles = set(collab_recs['title'].tolist()) if not collab_recs.empty else set()
            
            # Find overlapping recommendations
            common_recs = content_titles.intersection(collab_titles)
            
            # Create a Venn diagram
            if content_titles or collab_titles:
                from matplotlib_venn import venn2
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 6))
                venn = venn2(
                    [content_titles, collab_titles], 
                    ('Content-Based', 'Collaborative Filtering'),
                    ax=ax
                )
                
                st.pyplot(fig)
                
                if common_recs:
                    st.subheader("Movies Recommended by Both Methods")
                    st.write("These movies are highly recommended across methods:")
                    
                    for title in common_recs:
                        # Get the genres from the movies dataframe
                        genres = df_movies.loc[df_movies['title'] == title, 'genres'].iloc[0]
                        display_movie_card(title, genres=genres)
                        st.markdown("---")
                else:
                    st.info("No overlapping recommendations between methods.")
            else:
                st.info("Not enough data to compare recommendation methods.")
    else:
        st.info("Please select a movie to compare recommendation methods.")

def run_app():
    """Main function to run the Streamlit app"""
    # Set up page config
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Define pages
    PAGES = {
        "Home": home_page,
        "Recommendations": recommendations_page,
        "Content-Based Recommendations": content_based_recommendations_page,
        "Collaborative Filtering": item_based_recommendations_page,
        "Compare Methods": compare_recommendations_page
    }
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Home"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🎬 Movie Matcher")
        st.caption("Find your next favorite movie")
        
        # Navigation buttons
        for page_name in PAGES.keys():
            if st.button(
                page_name, 
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if st.session_state['current_page'] == page_name else "secondary"
            ):
                st.session_state['current_page'] = page_name
                st.rerun()
        
        # Add some information at the bottom of the sidebar
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This app uses machine learning to recommend movies based on your profile "
            "and preferences. Try different recommendation methods to find your perfect movie match!"
        )
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.caption("© 2025 Movie Recommendation System")
    
    # Display the selected page
    PAGES[st.session_state['current_page']]()

if __name__ == "__main__":
    run_app()
