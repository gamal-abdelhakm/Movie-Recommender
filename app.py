import imdb
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uszipcode import SearchEngine
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache functions to improve performance
@st.cache_data
def load_movies_dataset():
    """Load and cache movies dataset"""
    try:
        df_movies = pd.read_csv("merged.csv")
        return df_movies
    except FileNotFoundError:
        st.error("Movies dataset not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_ratings_dataset():
    """Load and cache ratings dataset"""
    try:
        df_ratings = pd.read_csv('ratings.csv', sep=';')
        return df_ratings
    except FileNotFoundError:
        st.error("Ratings dataset not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_users_dataset():
    """Load and cache users dataset"""
    try:
        df_users = pd.read_csv('users.csv', sep=';')
        return df_users
    except FileNotFoundError:
        st.error("Users dataset not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def load_movies_data():
    """Load and cache the movies data"""
    try:
        df_movies = pd.read_csv('movies.csv', sep=';', encoding='ISO-8859-1')
        if 'Unnamed: 3' in df_movies.columns:
            df_movies = df_movies.drop(['Unnamed: 3'], axis=1)
        return df_movies
    except FileNotFoundError:
        st.error("Movies data file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def get_state(zipcode):
    """Get state from zipcode using uszipcode"""
    try:
        search = SearchEngine()
        result = search.by_zipcode(zipcode)
        if result:
            return result.state
        else:
            return "Unknown"
    except Exception as e:
        st.error(f"Error getting state for zipcode: {e}")
        return "Unknown"

@st.cache_data
def get_movie_info(movie_title):
    """Get movie information from IMDb"""
    try:
        ia = imdb.IMDb()
        movies = ia.search_movie(movie_title)
        
        if movies:
            movie = movies[0]  # Get the first result
            ia.update(movie)  # Fetch additional information
            
            info = {
                'title': movie.get('title', 'N/A'),
                'year': movie.get('year', 'N/A'),
                'rating': movie.get('rating', 'N/A'),
                'genres': ', '.join(movie.get('genres', [])),
                'plot': movie.get('plot outline', 'Plot not available'),
                'director': ', '.join([d.get('name', 'N/A') for d in movie.get('directors', [])]),
                'cast': ', '.join([a.get('name', 'N/A') for a in movie.get('cast', [])[:3]]),
                'poster': movie.get('full-size cover url', movie.get('cover url', None))
            }
            return info
        return None
    except Exception as e:
        st.warning(f"Error fetching movie info: {str(e)}")
        return None

def preprocess_data(gender, age, occupation, zipcode):
    """Preprocess user data for cluster prediction"""
    try:
        # Convert gender to binary
        gender_binary = 0 if gender == 'Male' else 1
        
        # Get state from zipcode
        state = get_state(zipcode)
        
        # Load state encoder
        try:
            state_encoder = joblib.load("state_Encoder.h5")
            state_encoded = state_encoder.transform([state])
        except (FileNotFoundError, joblib.exceptions.JoblibException):
            st.error("State encoder model not found or error loading it.")
            return None
            
        # Return preprocessed data
        return [[gender_binary, age, occupation, state_encoded[0]]]
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def predict_cluster(gender, age, occupation, zipcode):
    """Predict user cluster based on input data"""
    try:
        # Preprocess user data
        user_data = preprocess_data(gender, age, occupation, zipcode)
        
        if user_data is None:
            return -1
            
        # Load KMeans model
        try:
            kmeans_model = joblib.load("cluster.h5")
            # Predict cluster
            cluster_number = kmeans_model.predict(user_data)[0]
            return cluster_number
        except (FileNotFoundError, joblib.exceptions.JoblibException):
            st.error("Cluster model not found or error loading it.")
            return -1
    except Exception as e:
        st.error(f"Error predicting cluster: {e}")
        return -1

def get_similar_movies(selected_movie, top_n=10):
    """Get content-based similar movies based on genres"""
    try:
        df_movies = load_movies_data()
        
        if df_movies.empty or selected_movie not in df_movies['title'].values:
            return pd.DataFrame()
            
        # Get selected movie genres
        selected_movie_genres = df_movies.loc[df_movies['title'] == selected_movie, 'genres'].iloc[0].split('|')
        
        # Find similar movies based on genre overlap
        similar_movies = df_movies[df_movies['genres'].apply(
            lambda x: any(genre in x.split('|') for genre in selected_movie_genres))].copy()
            
        # Calculate similarity score
        similar_movies['similarity_score'] = similar_movies['genres'].apply(
            lambda x: len(set(x.split('|')).intersection(selected_movie_genres)) / 
                  len(set(x.split('|')).union(selected_movie_genres)))
                  
        # Sort by similarity and exclude the input movie
        similar_movies = similar_movies.sort_values('similarity_score', ascending=False)
        similar_movies = similar_movies[similar_movies['title'] != selected_movie]
        
        # Return top N similar movies
        return similar_movies.head(top_n)[['title', 'genres', 'similarity_score']]
    except Exception as e:
        st.error(f"Error finding similar movies: {e}")
        return pd.DataFrame()

def get_item_based_recommendations(favorite_movie, top_n=10):
    """Get item-based collaborative filtering recommendations"""
    try:
        df_movies = load_movies_data()
        df_ratings = load_ratings_dataset()
        
        if df_movies.empty or df_ratings.empty or favorite_movie not in df_movies['title'].values:
            return pd.DataFrame()
            
        # Create user-item matrix
        user_item_matrix = df_ratings.pivot_table(
            index='movieId', 
            columns='userId', 
            values='rating'
        ).fillna(0)
        
        # Calculate item similarities
        movie_similarities = cosine_similarity(user_item_matrix)
        
        # Get index of the favorite movie
        movie_id = df_movies[df_movies['title'] == favorite_movie]['movieId'].iloc[0]
        movie_index = user_item_matrix.index.get_loc(movie_id) if movie_id in user_item_matrix.index else -1
        
        if movie_index == -1:
            return pd.DataFrame()
            
        # Get similar movies indices
        similar_indices = movie_similarities[movie_index].argsort()[::-1][1:top_n+1]
        similar_movie_ids = user_item_matrix.index[similar_indices]
        
        # Get similar movies details
        similar_movies = df_movies[df_movies['movieId'].isin(similar_movie_ids)][['movieId', 'title', 'genres']]
        similar_movies['similarity_score'] = [movie_similarities[movie_index][idx] for idx in similar_indices]
        
        return similar_movies.sort_values('similarity_score', ascending=False)
    except Exception as e:
        st.error(f"Error getting item-based recommendations: {e}")
        return pd.DataFrame()

def get_user_based_recommendations(cluster_number, top_n=5):
    """Get user-based recommendations for a specific cluster"""
    try:
        df_movies = load_movies_dataset()
        
        if df_movies.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        # Filter movies by cluster
        clustered_df = df_movies[df_movies['cluster'] == cluster_number]
        
        if clustered_df.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        # Get popular movies in the cluster
        popular_movies = clustered_df['title'].value_counts().head(top_n)
        popular_df = pd.DataFrame({'title': popular_movies.index, 'count': popular_movies.values})
        
        # Get highest rated movies in the cluster
        avg_ratings = clustered_df.groupby('title')['rating'].agg(['mean', 'count'])
        # Filter by minimum count of ratings to ensure reliability
        min_count = 5
        high_rated = avg_ratings[avg_ratings['count'] >= min_count].sort_values('mean', ascending=False)
        high_rated_df = pd.DataFrame({
            'title': high_rated.index[:top_n],
            'avg_rating': high_rated['mean'][:top_n].round(2),
            'rating_count': high_rated['count'][:top_n]
        })
        
        return popular_df, high_rated_df
    except Exception as e:
        st.error(f"Error getting user-based recommendations: {e}")
        return pd.DataFrame(), pd.DataFrame()

def display_movie_card(movie_info, col):
    """Display a movie card with image and details"""
    with col:
        if movie_info:
            # Display poster if available
            if movie_info.get('poster'):
                try:
                    response = requests.get(movie_info['poster'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=200)
                except Exception:
                    st.image("https://via.placeholder.com/200x300?text=No+Image", width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Image", width=200)
                
            # Display movie details
            st.markdown(f"### {movie_info['title']} ({movie_info.get('year', 'N/A')})")
            st.markdown(f"**IMDb Rating:** {movie_info.get('rating', 'N/A')}")
            st.markdown(f"**Genres:** {movie_info.get('genres', 'N/A')}")
            st.markdown(f"**Director:** {movie_info.get('director', 'N/A')}")
            
            with st.expander("More Details"):
                st.markdown(f"**Cast:** {movie_info.get('cast', 'N/A')}")
                st.markdown(f"**Plot:** {movie_info.get('plot', 'Plot not available')}")
        else:
            st.warning("Movie information not available")

def analyze_cluster(cluster_number):
    """Analyze cluster characteristics"""
    try:
        df_movies = load_movies_dataset()
        
        if df_movies.empty:
            return
            
        # Filter by cluster
        cluster_data = df_movies[df_movies['cluster'] == cluster_number]
        
        if cluster_data.empty:
            st.warning(f"No data available for cluster {cluster_number}")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre distribution
            st.subheader("Genre Distribution")
            all_genres = []
            for genres in cluster_data['genres'].str.split('|'):
                if isinstance(genres, list):
                    all_genres.extend(genres)
            
            genre_counts = pd.Series(all_genres).value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis', ax=ax)
            ax.set_title(f"Top Genres in Cluster {cluster_number}")
            ax.set_xlabel("Count")
            st.pyplot(fig)
            
        with col2:
            # Rating distribution
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(cluster_data['rating'], bins=10, kde=True, ax=ax)
            ax.set_title(f"Rating Distribution in Cluster {cluster_number}")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
        # Cluster statistics
        st.subheader("Cluster Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Average Rating", f"{cluster_data['rating'].mean():.2f}")
            
        with stats_col2:
            st.metric("Number of Movies", f"{len(cluster_data)}")
            
        with stats_col3:
            st.metric("Most Common Year", f"{cluster_data['year'].mode()[0]}")
            
    except Exception as e:
        st.error(f"Error analyzing cluster: {e}")

def main():
    """Main function for user data clustering page"""
    st.title("User Data Clustering")
    st.write("Enter your demographic information to find your movie preference cluster.")
    
    # Load occupations
    occupations_list = ["Accountant","Actor","Architect","Artist","Astronaut","Athlete","Author","Baker","Banker","Barista","Bartender","Biologist","Butcher","Carpenter","Chef","Chemist","Civil engineer","Dentist","Doctor","Electrician","Engineer","Firefighter","Flight attendant","Graphic designer","Hairdresser","Journalist","Lawyer","Librarian","Mechanic","Musician","Nurse","Pharmacist","Photographer","Pilot","Police officer","Professor","Programmer","Psychologist","Scientist","Software developer","Teacher","Translator","Veterinarian","Waiter/Waitress","Web developer","Writer"]
    
    # User input form
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", min_value=18, max_value=100, value=30)
            
        with col2:
            occupation = st.selectbox("Occupation", occupations_list)
            occupation_idx = occupations_list.index(occupation)
            
            # Load user dataset for zipcodes
            df_users = load_users_dataset()
            unique_zipcodes = sorted(df_users['zip-code'].unique()) if not df_users.empty else ["10001"]
            zipcode = st.selectbox("ZIP Code", unique_zipcodes)
        
        submit_button = st.form_submit_button("Find My Cluster")
    
    # Process form submission
    if submit_button:
        with st.spinner("Analyzing your profile..."):
            cluster_number = predict_cluster(gender, age, occupation_idx, zipcode)
            
            if cluster_number >= 0:
                st.success(f"You belong to Cluster {cluster_number}!")
                st.session_state.cluster = cluster_number
                
                # Show cluster analysis button
                if st.button("Analyze My Cluster"):
                    st.session_state.page = "cluster_analysis"
                    st.experimental_rerun()
                
                # Show recommendations button
                if st.button("Get Movie Recommendations"):
                    st.session_state.page = "recommendations"
                    st.experimental_rerun()
            else:
                st.error("Unable to predict cluster. Please check your input or try again.")

def display_recommendations():
    """Display movie recommendations for the user's cluster"""
    st.title("Your Movie Recommendations")
    
    if 'cluster' not in st.session_state:
        st.warning("Please find your cluster first!")
        if st.button("Go to Clustering"):
            st.session_state.page = "main"
            st.experimental_rerun()
        return
    
    cluster_number = st.session_state.cluster
    st.subheader(f"Recommendations for Cluster {cluster_number}")
    
    # Get recommendations
    popular_movies, top_rated_movies = get_user_based_recommendations(cluster_number)
    
    if popular_movies.empty and top_rated_movies.empty:
        st.warning("No recommendations available for this cluster.")
        return
    
    # Display popular movies
    st.markdown("## Most Popular Movies in Your Cluster")
    if not popular_movies.empty:
        cols = st.columns(min(5, len(popular_movies)))
        for i, (_, row) in enumerate(popular_movies.iterrows()):
            if i < len(cols):
                movie_info = get_movie_info(row['title'])
                display_movie_card(movie_info, cols[i])
    else:
        st.info("No popular movies found for this cluster.")
    
    # Display top rated movies
    st.markdown("## Highest Rated Movies in Your Cluster")
    if not top_rated_movies.empty:
        cols = st.columns(min(5, len(top_rated_movies)))
        for i, (_, row) in enumerate(top_rated_movies.iterrows()):
            if i < len(cols):
                with cols[i]:
                    movie_info = get_movie_info(row['title'])
                    if movie_info:
                        display_movie_card(movie_info, cols[i])
                        st.metric("Avg Rating", f"{row['avg_rating']}")
                        st.caption(f"Based on {row['rating_count']} ratings")
    else:
        st.info("No highly rated movies found for this cluster.")
    
    # Back button
    if st.button("Back to Profile"):
        st.session_state.page = "main"
        st.experimental_rerun()

def content_based_recommendations():
    """Content-based movie recommendations page"""
    st.title("Content-Based Movie Recommendations")
    st.write("Get movie recommendations based on genre similarity.")
    
    # Load movies data
    df_movies = load_movies_data()
    
    if df_movies.empty:
        st.error("Movie data not available.")
        return
    
    # Select favorite movie
    favorite_movie = st.selectbox(
        "Select Your Favorite Movie", 
        sorted(df_movies['title'].unique()),
        index=0,
        key="content_movie_select"
    )
    
    if st.button("Get Recommendations", key="content_rec_button"):
        with st.spinner("Finding similar movies..."):
            # Get similar movies
            similar_movies = get_similar_movies(favorite_movie, top_n=10)
            
            if similar_movies.empty:
                st.warning("No similar movies found.")
                return
            
            # Display results
            st.subheader(f"Movies Similar to '{favorite_movie}'")
            
            # Get genre information for the selected movie
            selected_movie_genres = df_movies[df_movies['title'] == favorite_movie]['genres'].iloc[0]
            st.info(f"Selected movie genres: {selected_movie_genres.replace('|', ', ')}")
            
            # Display movie cards in a grid
            cols = st.columns(5)
            for i, (_, row) in enumerate(similar_movies.iterrows()):
                col_idx = i % 5
                movie_info = get_movie_info(row['title'])
                
                with cols[col_idx]:
                    if movie_info:
                        # Display poster if available
                        if movie_info.get('poster'):
                            try:
                                response = requests.get(movie_info['poster'])
                                img = Image.open(BytesIO(response.content))
                                st.image(img, width=150)
                            except Exception:
                                st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                        else:
                            st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                            
                        st.markdown(f"**{movie_info['title']}**")
                        st.caption(f"Similarity: {row['similarity_score']:.2f}")
                        st.caption(f"Genres: {row['genres'].replace('|', ', ')}")
                        
                        # Show more details in expander
                        with st.expander("Details"):
                            st.markdown(f"**Year:** {movie_info.get('year', 'N/A')}")
                            st.markdown(f"**IMDb:** {movie_info.get('rating', 'N/A')}")
                            st.markdown(f"**Director:** {movie_info.get('director', 'N/A')}")
                    else:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Similarity: {row['similarity_score']:.2f}")
                        st.caption(f"Genres: {row['genres'].replace('|', ', ')}")

def item_based_recommendations():
    """Item-based collaborative filtering recommendations page"""
    st.title("Item-Based Collaborative Filtering")
    st.write("Get recommendations based on what other similar users liked.")
    
    # Load movies data
    df_movies = load_movies_data()
    
    if df_movies.empty:
        st.error("Movie data not available.")
        return
    
    # Select favorite movie
    favorite_movie = st.selectbox(
        "Select Your Favorite Movie", 
        sorted(df_movies['title'].unique()),
        index=0,
        key="collab_movie_select"
    )
    
    if st.button("Get Recommendations", key="collab_rec_button"):
        with st.spinner("Finding recommendations..."):
            # Get similar movies
            similar_movies = get_item_based_recommendations(favorite_movie, top_n=10)
            
            if similar_movies.empty:
                st.warning("No recommendations found. This movie might not have enough ratings.")
                return
            
            # Display results
            st.subheader(f"Users Who Liked '{favorite_movie}' Also Liked:")
            
            # Display movie cards in a grid
            cols = st.columns(5)
            for i, (_, row) in enumerate(similar_movies.iterrows()):
                col_idx = i % 5
                movie_info = get_movie_info(row['title'])
                
                with cols[col_idx]:
                    if movie_info:
                        # Display poster if available
                        if movie_info.get('poster'):
                            try:
                                response = requests.get(movie_info['poster'])
                                img = Image.open(BytesIO(response.content))
                                st.image(img, width=150)
                            except Exception:
                                st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                        else:
                            st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                            
                        st.markdown(f"**{movie_info['title']}**")
                        st.caption(f"Correlation: {row['similarity_score']:.2f}")
                        st.caption(f"Genres: {row['genres'].replace('|', ', ')}")
                        
                        # Show more details in expander
                        with st.expander("Details"):
                            st.markdown(f"**Year:** {movie_info.get('year', 'N/A')}")
                            st.markdown(f"**IMDb:** {movie_info.get('rating', 'N/A')}")
                            st.markdown(f"**Director:** {movie_info.get('director', 'N/A')}")
                    else:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Correlation: {row['similarity_score']:.2f}")
                        st.caption(f"Genres: {row['genres'].replace('|', ', ')}")

def cluster_analysis():
    """Cluster analysis page"""
    st.title("Cluster Analysis Dashboard")
    
    if 'cluster' not in st.session_state:
        st.warning("Please find your cluster first!")
        if st.button("Go to Clustering"):
            st.session_state.page = "main"
            st.experimental_rerun()
        return
    
    cluster_number = st.session_state.cluster
    st.subheader(f"Analysis for Cluster {cluster_number}")
    
    # Display cluster analysis
    analyze_cluster(cluster_number)
    
    # Back button
    if st.button("Back to Profile"):
        st.session_state.page = "main"
        st.experimental_rerun()

def about_page():
    """About page with information about the system"""
    st.title("About this Movie Recommendation System")
    
    st.markdown("""
    ### Overview
    This application combines multiple recommendation approaches to help you discover movies you'll love:
    
    1. **User Clustering** - Groups users with similar demographics and preferences
    2. **Content-Based Filtering** - Recommends movies similar to ones you already like
    3. **Collaborative Filtering** - Suggests movies based on what similar users enjoyed
    
    ### How it Works
    
    #### User Clustering
    The system uses K-means clustering to group users with similar characteristics and movie preferences. 
    By entering your demographic information, we can match you to a cluster of users with similar tastes.
    
    #### Content-Based Recommendations
    This approach analyzes movie features (primarily genres) to find similar movies. 
    If you enjoy action movies with sci-fi elements, we'll find other movies that share those characteristics.
    
    #### Collaborative Filtering
    This method identifies patterns in user ratings to find movies you might enjoy. 
    It works on the principle that users who agreed in the past will likely agree in the future.
    
    ### Technologies Used
    - **Python** - Core programming language
    - **Streamlit** - Web application framework
    - **Pandas** - Data processing and analysis
    - **Scikit-learn** - Machine learning algorithms
    - **IMDb API** - Movie information and posters
    - **Matplotlib & Seaborn** - Data visualization
    
    ### Feedback
    We're constantly improving our recommendation algorithms. Your feedback helps us make better suggestions!
    """)

# Define page navigation
PAGES = {
    "User Clustering": main,
    "Movie Recommendations": display_recommendations,
    "Content-Based Recommendations": content_based_recommendations,
    "Collaborative Filtering": item_based_recommendations,
    "Cluster Analysis": cluster_analysis,
    "About": about_page
}

def run_app():
    """Main application function"""
    # Custom CSS
    st.markdown("""
    <style>
    .movie-title {
        font-weight: bold;
        margin-bottom: 0;
    }
    .movie-info {
        font-size: 0.9em;
        color: #666;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    # Sidebar navigation
    st.sidebar.title("🎬 Movie Recommender")
    st.sidebar.markdown("---")
    
    # Navigation menu
    selected_page = st.sidebar.radio("Navigation", list(PAGES.keys()))
    
    # Display selected page
    PAGES[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Movie Recommendation System © 2025")

if __name__ == "__main__":
    run_app()
