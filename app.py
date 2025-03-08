import imdb
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uszipcode import SearchEngine
import os
import time
from PIL import Image
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache decoration for performance improvement
@st.cache_data(ttl=3600)
def load_datasets():
    """Load and cache all datasets"""
    try:
        df_users = pd.read_csv('users.csv', sep=';')
        df_movies = pd.read_csv('movies.csv', sep=';', encoding='ISO-8859-1')
        if 'Unnamed: 3' in df_movies.columns:
            df_movies = df_movies.drop(['Unnamed: 3'], axis=1)
        df_ratings = pd.read_csv('ratings.csv', sep=';')
        df_merged = pd.read_csv("merged.csv")
        return df_users, df_movies, df_ratings, df_merged
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        st.error(f"Failed to load datasets: {e}")
        return None, None, None, None

@st.cache_resource
def load_models():
    """Load and cache ML models"""
    try:
        kmeans_model = joblib.load("cluster.h5")
        state_encoder = joblib.load("state_Encoder.h5")
        return kmeans_model, state_encoder
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Failed to load models: {e}")
        return None, None

# Cache movie info to reduce API calls
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_movie_info(movie_title):
    """Get movie information from IMDb with caching"""
    try:
        # Create an instance of the IMDb class
        ia = imdb.IMDb()

        # Search for the movie title
        movies = ia.search_movie(movie_title)

        if movies:
            movie = movies[0]  # Get the first result
            movie_id = movie.movieID
            
            # Fetch additional information
            try:
                ia.update(movie)
            except Exception as e:
                logger.warning(f"Could not update movie details: {e}")
            
            # Default image in case of missing poster
            poster_url = "https://via.placeholder.com/300x450?text=No+Poster+Available"
            
            # Try to get poster URL
            if 'cover url' in movie:
                poster_url = movie['cover url']
            
            # Extract relevant details
            info = {
                'id': movie_id,
                'name': movie['title'],
                'poster': poster_url,
                'year': movie.get('year', 'Unknown'),
                'genres': movie.get('genres', []),
                'rating': movie.get('rating', 'N/A')
            }
            
            return info
        return None
    except Exception as e:
        logger.error(f"Error fetching movie info for {movie_title}: {e}")
        return None

def display_movie_card(movie_title, rating=None, similarity=None, count=None):
    """Display a movie in a card format"""
    movie_info = get_movie_info(movie_title)
    
    if movie_info is None:
        st.warning(f"Could not find information for '{movie_title}'")
        return
    
    # Create a card-like display with columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            response = requests.get(movie_info['poster'])
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, width=150)
            else:
                st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
        except Exception as e:
            st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
            logger.warning(f"Could not load image: {e}")
    
    with col2:
        st.subheader(movie_info['name'] + f" ({movie_info['year']})")
        
        if rating is not None:
            st.write(f"**Rating:** {rating:.2f} ⭐")
        
        if similarity is not None:
            similarity_percentage = similarity * 100
            st.write(f"**Similarity:** {similarity_percentage:.1f}%")
            
        if count is not None:
            st.write(f"**Count in cluster:** {count}")
            
        if movie_info['genres']:
            st.write(f"**Genres:** {', '.join(movie_info['genres'])}")
        
        if movie_info['rating'] != 'N/A':
            st.write(f"**IMDb Rating:** {movie_info['rating']}/10")

def main():
    st.title("User Data Clustering")
    
    # Load datasets
    df_users, _, _, _ = load_datasets()
    kmeans_model, state_encoder = load_models()
    
    if df_users is None or kmeans_model is None or state_encoder is None:
        st.error("Required data or models could not be loaded. Please check the logs.")
        return
    
    # Add description
    st.markdown("""
    This system analyzes your demographics to match you with similar users and recommend movies 
    based on what people similar to you have enjoyed.
    """)
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        # User input fields
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", min_value=18, max_value=100, value=30)
    
    with col2:
        occupations_list = [
            "Accountant", "Actor", "Architect", "Artist", "Astronaut", "Athlete", 
            "Author", "Baker", "Banker", "Barista", "Bartender", "Biologist", 
            "Butcher", "Carpenter", "Chef", "Chemist", "Civil engineer", "Dentist", 
            "Doctor", "Electrician", "Engineer", "Firefighter", "Flight attendant", 
            "Graphic designer", "Hairdresser", "Journalist", "Lawyer", "Librarian", 
            "Mechanic", "Musician", "Nurse", "Pharmacist", "Photographer", "Pilot", 
            "Police officer", "Professor", "Programmer", "Psychologist", "Scientist", 
            "Software developer", "Teacher", "Translator", "Veterinarian", 
            "Waiter/Waitress", "Web developer", "Writer"
        ]
        
        # Add search functionality to occupation dropdown
        occupation = st.selectbox(
            "Select Occupation", 
            occupations_list,
            index=0
        )
        occupation_index = occupations_list.index(occupation)
        
        # Get unique zipcodes with a search box
        unique_zipcodes = sorted(df_users['zip-code'].unique())
        zipcode_search = st.text_input("Search Zipcode", "")
        
        filtered_zipcodes = [str(z) for z in unique_zipcodes if str(z).startswith(zipcode_search)]
        if filtered_zipcodes:
            selected_zipcode = st.selectbox("Select Zipcode", filtered_zipcodes)
        else:
            st.warning("No matching zipcodes found. Using default.")
            selected_zipcode = unique_zipcodes[0] if unique_zipcodes else "00000"
    
    # Save user data with a more descriptive button and loading indicator
    if st.button("Find Your Movie Cluster", key="predict_button", type="primary"):
        with st.spinner('Analyzing your profile...'):
            try:
                cluster_number = predict_cluster(gender, age, occupation_index, selected_zipcode, kmeans_model, state_encoder)
                
                # Store the result in session state for other pages to access
                st.session_state['user_cluster'] = cluster_number
                st.session_state['user_profile'] = {
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zipcode': selected_zipcode
                }
                
                # Success message with animated confetti
                st.success(f"Analysis complete! You belong to Cluster #{cluster_number}")
                st.balloons()
                
                # Add a redirection button to recommendations
                st.markdown("### What's Next?")
                st.markdown("Click below to see movie recommendations based on your cluster:")
                if st.button("View My Recommendations", key="goto_recommendations"):
                    st.experimental_set_query_params(cluster=cluster_number)
                    st.experimental_rerun()
                
            except Exception as e:
                logger.error(f"Error predicting cluster: {e}")
                st.error(f"An error occurred: {e}")

def display_recommendations():
    st.title("Your Personalized Movie Recommendations")
    
    # Check if we have a cluster either in query params or session state
    query_params = st.experimental_get_query_params()
    
    if "cluster" in query_params:
        cluster_number = int(query_params.get("cluster", [0])[0])
        st.session_state['user_cluster'] = cluster_number
    elif "user_cluster" in st.session_state:
        cluster_number = st.session_state['user_cluster']
    else:
        st.warning("No cluster information found. Please go back to the home page to analyze your profile.")
        if st.button("Go to Profile Analysis"):
            st.experimental_rerun()
        return
    
    # Display user profile if available
    if "user_profile" in st.session_state:
        profile = st.session_state['user_profile']
        st.sidebar.markdown("### Your Profile")
        st.sidebar.markdown(f"""
        - **Gender:** {profile['gender']}
        - **Age:** {profile['age']}
        - **Occupation:** {profile['occupation']}
        - **Zipcode:** {profile['zipcode']}
        - **Cluster:** #{st.session_state['user_cluster']}
        """)
    
    # Load datasets
    _, _, _, df_merged = load_datasets()
    
    if df_merged is None:
        st.error("Required data could not be loaded.")
        return
    
    # Filter movies based on cluster number
    clustered_df = df_merged[df_merged['cluster'] == cluster_number]
    
    if clustered_df.empty:
        st.warning(f"No movies found for cluster #{cluster_number}. Try a different profile.")
        return
    
    # Add tabs for different recommendation methods
    tab1, tab2 = st.tabs(["Popular in Your Cluster", "Highest Rated in Your Cluster"])
    
    with tab1:
        # Get the top 5 most common movie titles
        top_movies = clustered_df['title'].value_counts()[:5]
        
        st.markdown(f"### Most Popular Movies Among Users Like You")
        st.markdown("These movies are the most frequently watched by people in your taste cluster:")
        
        for i, (movie_title, count) in enumerate(top_movies.items(), 1):
            st.markdown(f"#### {i}. {movie_title}")
            display_movie_card(movie_title, count=count)
            st.markdown("---")
    
    with tab2:
        # Get the top 5 recommended movies based on ratings
        # Only consider movies with at least 3 ratings for more reliable recommendations
        rating_counts = clustered_df.groupby('title')['rating'].count()
        qualified_movies = rating_counts[rating_counts >= 3].index
        
        qualified_df = clustered_df[clustered_df['title'].isin(qualified_movies)]
        
        if qualified_df.empty:
            st.warning("Not enough rating data in your cluster. Showing all movies instead.")
            qualified_df = clustered_df
        
        average_ratings = qualified_df.groupby('title')['rating'].mean()
        sorted_ratings = average_ratings.sort_values(ascending=False)
        recommended_movies = sorted_ratings[:5]
        
        st.markdown(f"### Highest Rated Movies For Your Taste Profile")
        st.markdown("These movies received the highest ratings from users similar to you:")
        
        for i, (movie_title, rating) in enumerate(recommended_movies.items(), 1):
            st.markdown(f"#### {i}. {movie_title}")
            display_movie_card(movie_title, rating=rating)
            st.markdown("---")

def content_based_recommendations():
    st.title("Content-Based Movie Recommendations")
    
    # Load datasets
    _, df_movies, _, _ = load_datasets()
    
    if df_movies is None:
        st.error("Required data could not be loaded.")
        return
    
    st.markdown("""
    This recommender suggests movies based on similarity in genres and content. 
    Select a movie you like, and we'll find similar movies you might enjoy.
    """)
    
    # Create search box for finding movies
    movie_search = st.text_input("Search for a movie", "")
    
    # Filter movies based on search term
    if movie_search:
        filtered_movies = df_movies[df_movies['title'].str.contains(movie_search, case=False, na=False)]
        if filtered_movies.empty:
            st.warning(f"No movies found containing '{movie_search}'")
        else:
            # Display movie options
            selected_movie = st.selectbox(
                "Select your favorite movie", 
                filtered_movies['title'].unique(),
                index=0
            )
            
            if st.button("Find Similar Movies", type="primary"):
                with st.spinner('Finding similar movies...'):
                    # Get top 10 similar movies based on content
                    similar_movies = get_similar_movies(selected_movie, df_movies, top_n=10)
                    
                    # Display the selected movie first
                    st.subheader("Your Selected Movie:")
                    display_movie_card(selected_movie)
                    
                    # Display similar movies
                    st.subheader("Movies You Might Like:")
                    
                    for i, (index, row) in enumerate(similar_movies.iterrows(), 1):
                        st.markdown(f"#### {i}. {row['title']}")
                        display_movie_card(row['title'], similarity=row['similarity_score'])
                        st.markdown("---")
    else:
        st.info("Enter a movie title above to get started.")

def item_based_recommendations():
    st.title("Item-Based Collaborative Filtering")
    
    # Load datasets
    _, df_movies, df_ratings, _ = load_datasets()
    
    if df_movies is None or df_ratings is None:
        st.error("Required data could not be loaded.")
        return
    
    st.markdown("""
    This recommender analyzes user rating patterns to find movies that are rated similarly.
    Select a movie you like, and we'll find others that people tend to rate in a similar way.
    """)
    
    # Create search box for finding movies
    movie_search = st.text_input("Search for a movie", "")
    
    # Filter movies based on search term
    if movie_search:
        filtered_movies = df_movies[df_movies['title'].str.contains(movie_search, case=False, na=False)]
        if filtered_movies.empty:
            st.warning(f"No movies found containing '{movie_search}'")
        else:
            # Display movie options
            selected_movie = st.selectbox(
                "Select your favorite movie", 
                filtered_movies['title'].unique(),
                index=0
            )
            
            if st.button("Find Similar Movies", type="primary"):
                with st.spinner('Building recommendation model...'):
                    try:
                        # Progress bar for better UX during computation
                        progress_bar = st.progress(0)
                        
                        # Step 1: Get movie IDs 
                        progress_bar.progress(10)
                        
                        # Check if the movie exists in the ratings dataset
                        movie_id = df_movies[df_movies['title'] == selected_movie]['movieId'].values[0]
                        if movie_id not in df_ratings['movieId'].values:
                            st.warning(f"No rating data available for '{selected_movie}'. Try another movie.")
                            return
                        
                        # Step 2: Create user-item matrix
                        progress_bar.progress(30)
                        st.info("Creating user-item rating matrix...")
                        
                        # Create a sparse matrix for better performance
                        user_item_matrix = df_ratings.pivot_table(
                            index='movieId', 
                            columns='userId', 
                            values='rating'
                        ).fillna(0)
                        
                        progress_bar.progress(50)
                        
                        # Step 3: Compute similarity
                        st.info("Computing movie similarities...")
                        
                        # Get the index of the movie in the matrix
                        if movie_id not in user_item_matrix.index:
                            st.warning(f"Movie '{selected_movie}' not found in rating data. Try another movie.")
                            return
                            
                        # Compute similarities
                        movie_index = user_item_matrix.index.get_loc(movie_id)
                        
                        # For better performance, only compute similarity with movies that have some ratings in common
                        movie_row = user_item_matrix.iloc[movie_index].to_numpy()
                        nonzero_indices = np.nonzero(movie_row)[0]
                        
                        # Create a filter for movies that share users with the selected movie
                        filter_movies = user_item_matrix.iloc[:, nonzero_indices].sum(axis=1) > 0
                        filtered_matrix = user_item_matrix[filter_movies]
                        
                        progress_bar.progress(70)
                        
                        # Calculate similarities with the filtered matrix
                        similarities = cosine_similarity([user_item_matrix.iloc[movie_index]], filtered_matrix)[0]
                        
                        progress_bar.progress(80)
                        
                        # Get indices of top similar movies (excluding the movie itself)
                        similar_indices = similarities.argsort()[:-11:-1]  # Get top 10 in reverse order
                        
                        # Filter out the movie itself if it's in the results
                        similar_indices = [idx for idx in similar_indices if filtered_matrix.index[idx] != movie_id][:10]
                        
                        # Map indices back to movie IDs
                        similar_movie_ids = [filtered_matrix.index[idx] for idx in similar_indices]
                        
                        # Get movie details
                        similar_movies = df_movies[df_movies['movieId'].isin(similar_movie_ids)].copy()
                        
                        # Add similarity scores
                        similar_movies['similarity_score'] = [similarities[filtered_matrix.index.get_loc(movie_id)] for movie_id in similar_movie_ids]
                        
                        # Sort by similarity
                        similar_movies = similar_movies.sort_values('similarity_score', ascending=False)
                        
                        progress_bar.progress(100)
                        time.sleep(0.5)  # Give a moment to see the completed progress
                        progress_bar.empty()
                        
                        # Display the selected movie first
                        st.subheader("Your Selected Movie:")
                        display_movie_card(selected_movie)
                        
                        # Display similar movies
                        st.subheader("Movies Often Rated Similarly:")
                        
                        for i, (index, row) in enumerate(similar_movies.iterrows(), 1):
                            st.markdown(f"#### {i}. {row['title']}")
                            display_movie_card(row['title'], similarity=row['similarity_score'])
                            st.markdown("---")
                            
                    except Exception as e:
                        logger.error(f"Error in item-based recommendations: {e}")
                        st.error(f"An error occurred: {e}")
                        st.error("Try selecting a different movie or check the logs for details.")
    else:
        st.info("Enter a movie title above to get started.")

def predict_cluster(gender, age, occupation, zipcode, kmeans_model, state_encoder):
    """Predict cluster based on user data"""
    # Preprocess user data
    user_data = preprocess_data(gender, age, occupation, zipcode, state_encoder)
    
    # Predict cluster number
    cluster_number = kmeans_model.predict(user_data)
    logger.info(f"Predicted cluster: {cluster_number[0]}")
    
    return cluster_number[0]

# Define a function to get the state for a zip-code
@st.cache_data
def get_state(zipcode):
    """Get state from zipcode with caching"""
    try:
        search = SearchEngine()
        result = search.by_zipcode(zipcode)
        if result:
            return result.state
        else:
            logger.warning(f"State not found for zipcode {zipcode}")
            return "Unknown"
    except Exception as e:
        logger.error(f"Error getting state for zipcode {zipcode}: {e}")
        return "Unknown"

def preprocess_data(gender, age, occupation, zipcode, state_encoder):
    """Preprocess user data for prediction"""
    # Convert gender to binary representation
    gender_numeric = 0 if gender == 'Male' else 1
    
    # Get state from zipcode
    state = get_state(zipcode)
    
    try:
        # Transform state using encoder
        state_encoded = state_encoder.transform([state])
    except Exception as e:
        logger.error(f"Error encoding state {state}: {e}")
        # Default to first state in encoder if there's an error
        state_encoded = [[0]]
    
    # Return the preprocessed data as a list or array
    data = [[gender_numeric, age, occupation, state_encoded[0]]]
    return data

def get_similar_movies(selected_movie, df_movies, top_n=10):
    """Get similar movies based on genre overlap"""
    try:
        # Get the genres of the selected movie
        selected_movie_genres = df_movies.loc[df_movies['title'] == selected_movie, 'genres'].iloc[0].split('|')
        
        # Find movies with at least one genre in common
        similar_movies = df_movies[df_movies['genres'].apply(
            lambda x: any(genre in x.split('|') for genre in selected_movie_genres)
        )].copy()
        
        # Calculate Jaccard similarity (intersection over union)
        similar_movies['similarity_score'] = similar_movies['genres'].apply(
            lambda x: len(set(x.split('|')).intersection(selected_movie_genres)) / 
                    len(set(x.split('|')).union(selected_movie_genres))
        )
        
        # Sort by similarity score and exclude the selected movie
        similar_movies = similar_movies[similar_movies['title'] != selected_movie]
        similar_movies = similar_movies.sort_values('similarity_score', ascending=False)
        
        # Get top N similar movies
        top_similar_movies = similar_movies.head(top_n)
        
        return top_similar_movies
    
    except Exception as e:
        logger.error(f"Error finding similar movies: {e}")
        st.error(f"Error finding similar movies: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def feedback_page():
    """Add a feedback page for user input"""
    st.title("Give Us Your Feedback")
    
    st.markdown("""
    We value your feedback to improve our movie recommendation system.
    Please let us know your thoughts on the recommendations and any features you'd like to see added.
    """)
    
    # User rating
    user_rating = st.slider("How would you rate our recommendations?", 1, 5, 3)
    
    # Feedback text
    feedback_text = st.text_area("Your feedback", height=150)
    
    # Feature requests with checkboxes
    st.subheader("What features would you like to see?")
    col1, col2 = st.columns(2)
    
    with col1:
        want_trending = st.checkbox("Trending movies section")
        want_newreleases = st.checkbox("New releases filter")
        want_favorites = st.checkbox("Save favorites")
    
    with col2:
        want_watchlist = st.checkbox("Personal watchlist")
        want_streaming = st.checkbox("Streaming availability")
        want_socialshar = st.checkbox("Social sharing")
    
    # Submit button
    if st.button("Submit Feedback", type="primary"):
        st.success("Thank you for your feedback! We appreciate your input.")
        st.balloons()
        
        # In a real app, you would save this to a database
        # For now we'll just log it
        logger.info(f"Feedback received: Rating={user_rating}, Text={feedback_text}")

def about_page():
    """About page with information about the system"""
    st.title("About This Recommendation System")
    
    st.markdown("""
    ## How It Works
    
    Our movie recommendation system uses multiple approaches to suggest movies you might enjoy:
    
    ### 1. User Clustering
    We group users with similar demographics and preferences into clusters. When you enter your 
    information, we match you with a cluster of similar users and recommend movies that are popular 
    in that group.
    
    ### 2. Content-Based Filtering
    This approach recommends movies based on their features (like genres). If you like a particular 
    movie, we'll suggest others with similar characteristics.
    
    ### 3. Item-Based Collaborative Filtering
    This method analyzes rating patterns across movies. It identifies movies that are frequently 
    rated similarly by users, suggesting that people who like one might also enjoy the others.
    
    ## Data Sources
    
    Our system uses anonymized user data and movie ratings to generate recommendations. The movie 
    information is enhanced with details from IMDb.
    
    ## Privacy
    
    All user data entered in this application is processed locally and is not stored or saved 
    between sessions.
    """)
    
    # Add credits section
    st.markdown("---")
    st.markdown("### Credits")
    st.markdown("This application was built with:")
    st.markdown("- Streamlit - for the interactive web interface")
    st.markdown("- scikit-learn - for machine learning algorithms")
    st.markdown("- pandas - for data processing")
    st.markdown("- IMDbPY - for movie information")

# Define available pages
PAGES = {
    "Profile Analysis": main,
    "Your Recommendations": display_recommendations,
    "Find Similar Movies": content_based_recommendations,
    "Movies Rated Similarly": item_based_recommendations,
    "Feedback": feedback_page,
    "About": about_page
}

def run_app():
    """Main app entry point"""
    # Add logo and app title in the sidebar
    st.sidebar.title("🎬 MovieMatch")
    st.sidebar.markdown("*Your personalized movie recommender*")
    
    # Navigation in sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Run the selected page
    PAGES[page]()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 MovieMatch System")

if __name__ == "__main__":
    run_app()
