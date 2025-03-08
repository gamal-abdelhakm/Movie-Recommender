import imdb
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uszipcode import SearchEngine
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Cache data loading functions for better performance
@st.cache_data
def load_movies_dataset():
    df_movies = pd.read_csv("merged.csv")
    return df_movies

@st.cache_data
def load_ratings_dataset():
    df_ratings = pd.read_csv('ratings.csv', sep=';')
    return df_ratings

@st.cache_data
def load_users_dataset():
    df_users = pd.read_csv('users.csv', sep=';')
    return df_users

@st.cache_data
def load_movies_for_content():
    df_movies = pd.read_csv('movies.csv', sep=';', encoding='ISO-8859-1')
    if 'Unnamed: 3' in df_movies.columns:
        df_movies = df_movies.drop(['Unnamed: 3'], axis=1)
    return df_movies

# Movie information retrieval with caching
@st.cache_data
def get_movie_info(movie_title):
    try:
        ia = imdb.IMDb()
        movies = ia.search_movie(movie_title)
        
        if movies:
            movie = movies[0]  # Get the first result
            ia.update(movie)  # Fetch additional information
            
            # Extract relevant details
            info = {
                'name': movie['title'],
                'poster': movie.get('cover url', None),
                'year': movie.get('year', 'Unknown'),
                'cast': ', '.join([actor['name'] for actor in movie.get('cast', [])[:3]]),
                'director': ', '.join([director['name'] for director in movie.get('directors', [])]),
                'rating': movie.get('rating', 'Unknown'),
                'plot': movie.get('plot outline', 'Plot not available')
            }
            return info
        return None
    except Exception as e:
        st.error(f"Error fetching movie info: {e}")
        return None

def predict_cluster(gender, age, occupation, zipcode):
    try:
        # Load the KMeans model
        kmeans_model = joblib.load("cluster.h5")
        
        # Preprocess user data
        user_data = preprocess_data(gender, age, occupation, zipcode)
        
        # Predict cluster number
        cluster_number = kmeans_model.predict(user_data)
        return int(cluster_number[0])
    except Exception as e:
        st.error(f"Error in cluster prediction: {e}")
        return 0

def preprocess_data(gender, age, occupation, zipcode):
    # Convert gender to binary representation
    gender_binary = 0 if gender == 'Male' else 1
    
    # Get state from zipcode
    state = get_state(zipcode)
    
    # Load state encoder
    state_encoder = joblib.load("state_Encoder.h5")
    state_encoded = state_encoder.transform([state])
    
    # Return the preprocessed data as a list or array
    return [[gender_binary, age, occupation, state_encoded[0]]]

def get_state(zipcode):
    search = SearchEngine()
    result = search.by_zipcode(zipcode)
    if result:
        return result.state
    else:
        return "Unknown"

def get_similar_movies(selected_movie, top_n=10):
    df_movies = load_movies_for_content()
    
    # Get genres of selected movie
    selected_movie_row = df_movies[df_movies['title'] == selected_movie]
    if selected_movie_row.empty:
        return pd.DataFrame(columns=['title', 'genres', 'similarity_score'])
    
    selected_movie_genres = selected_movie_row['genres'].iloc[0].split('|')
    
    # Find movies with similar genres - use a safer approach that doesn't rely on regex
    similar_movies = []
    for _, movie in df_movies.iterrows():
        if isinstance(movie['genres'], str):
            movie_genres = movie['genres'].split('|')
            # Calculate Jaccard similarity
            intersection = len(set(movie_genres).intersection(selected_movie_genres))
            union = len(set(movie_genres).union(selected_movie_genres))
            similarity = intersection / union if union > 0 else 0
            similar_movies.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity_score': similarity
            })
    
    # Convert to DataFrame and sort
    similar_movies_df = pd.DataFrame(similar_movies)
    similar_movies_df = similar_movies_df.sort_values('similarity_score', ascending=False)
    
    # Get top N similar movies (excluding the selected movie)
    top_similar_movies = similar_movies_df[similar_movies_df['title'] != selected_movie].head(top_n)
    
    return top_similar_movies

def get_item_based_recommendations(favorite_movie, top_n=10):
    df_movies = load_movies_for_content()
    df_ratings = load_ratings_dataset()
    
    # Create user-item matrix
    user_item_matrix = df_ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
    
    # Calculate item-item similarity
    movie_similarities = cosine_similarity(user_item_matrix)
    
    try:
        # Find index of favorite movie
        movie_index = df_movies[df_movies['title'] == favorite_movie].index[0]
        
        # Get similar movies
        similar_movies_indices = movie_similarities[movie_index].argsort()[::-1][1:top_n+1]
        similar_movies = df_movies.iloc[similar_movies_indices].copy()
        
        # Add similarity score
        similar_movies['similarity_score'] = [movie_similarities[movie_index][idx] for idx in similar_movies_indices]
        
        return similar_movies[['title', 'genres', 'similarity_score']]
    except (IndexError, KeyError):
        st.error(f"Could not find movie '{favorite_movie}' in the dataset")
        return pd.DataFrame(columns=['title', 'genres', 'similarity_score'])

def display_movie_card(movie_info, col, show_plot=True):
    with col:
        st.subheader(movie_info['name'] + f" ({movie_info.get('year', '')})")
        
        # Create two columns within the card
        img_col, details_col = st.columns([1, 2])
        
        with img_col:
            if movie_info.get('poster'):
                st.image(movie_info['poster'], use_container_width=True)
            else:
                st.image("https://via.placeholder.com/300x450?text=No+Poster+Available", use_container_width=True)
        
        with details_col:
            st.markdown(f"**IMDb Rating:** {movie_info.get('rating', 'N/A')}")
            st.markdown(f"**Director:** {movie_info.get('director', 'N/A')}")
            st.markdown(f"**Cast:** {movie_info.get('cast', 'N/A')}")
            
            if show_plot and movie_info.get('plot'):
                st.markdown("**Plot:**")
                st.markdown(f"_{movie_info.get('plot', 'Plot not available')}_")

def main():
    st.title("🎭 User Profile & Cluster Analysis")
    
    df_users = load_users_dataset()
    
    with st.form("user_profile_form"):
        st.subheader("Enter Your Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", min_value=18, max_value=100, value=30)
        
        with col2:
            occupations_list = ["Accountant", "Actor", "Architect", "Artist", "Astronaut", "Athlete", 
                               "Author", "Baker", "Banker", "Barista", "Bartender", "Biologist", 
                               "Butcher", "Carpenter", "Chef", "Chemist", "Civil engineer", "Dentist", 
                               "Doctor", "Electrician", "Engineer", "Firefighter", "Flight attendant", 
                               "Graphic designer", "Hairdresser", "Journalist", "Lawyer", "Librarian", 
                               "Mechanic", "Musician", "Nurse", "Pharmacist", "Photographer", "Pilot", 
                               "Police officer", "Professor", "Programmer", "Psychologist", "Scientist", 
                               "Software developer", "Teacher", "Translator", "Veterinarian", 
                               "Waiter/Waitress", "Web developer", "Writer"]
            
            occupation = st.selectbox("Occupation", occupations_list)
            occupation_idx = occupations_list.index(occupation)
            
            df_users['zip-code'] = df_users['zip-code'].astype(str).str.split('-').str[0]
            unique_zipcodes = sorted(df_users['zip-code'].unique())
            selected_zipcode = st.selectbox("Zipcode", unique_zipcodes)
        
        submitted = st.form_submit_button("Find My Movie Recommendations")
    
    if submitted:
        with st.spinner("Analyzing your profile..."):
            cluster_number = predict_cluster(gender, age, occupation_idx, selected_zipcode)
            
            st.session_state['user_cluster'] = cluster_number
            st.session_state['user_profile'] = {
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zipcode': selected_zipcode
            }
            
            st.success(f"🎉 Analysis complete! You belong to Cluster #{cluster_number}")
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Cluster Information", "Similar Users"])
            
            with tab1:
                display_cluster_info(cluster_number)
            
            with tab2:
                display_similar_users(gender, age, occupation, cluster_number)

def display_cluster_info(cluster_number):
    st.subheader(f"Cluster #{cluster_number} Profile")
    
    df_movies = load_movies_dataset()
    
    # Filter movies for the cluster
    clustered_df = df_movies[df_movies['cluster'] == cluster_number]
    
    if clustered_df.empty:
        st.warning("No data available for this cluster.")
        return
    
    # Create a layout with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Top genres in cluster
        st.subheader("Top Genres in Your Cluster")
        
        # Extract all genres from the cluster - safely without using regex
        all_genres = []
        for genres in clustered_df['genres'].dropna():
            if isinstance(genres, str):
                all_genres.extend(genres.split('|'))
        
        # Count genre occurrences
        genre_counts = Counter(all_genres)
        top_genres = dict(genre_counts.most_common(10))
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(list(top_genres.keys()), list(top_genres.values()), color='skyblue')
        ax.set_xlabel('Count')
        ax.set_title('Top 10 Genres in Your Cluster')
        
        # Add count labels to bars
        for i, v in enumerate(top_genres.values()):
            ax.text(v + 0.1, i, str(v), va='center')
            
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Average ratings by genre
        st.subheader("Average Ratings by Genre")
        
        # Calculate average rating per genre - safely without regex
        genre_ratings = {}
        for genre in genre_counts.keys():
            # This is the problematic part - need to avoid regex
            # genre_movies = clustered_df[clustered_df['genres'].str.contains(genre, na=False)]
            
            # Safer approach - use explicit string matching
            genre_movies = clustered_df[clustered_df['genres'].apply(
                lambda x: isinstance(x, str) and genre in x.split('|')
            )]
            
            if not genre_movies.empty:
                avg_rating = genre_movies['rating'].mean()
                genre_ratings[genre] = avg_rating
        
        # Sort and get top genres by rating
        sorted_genre_ratings = dict(sorted(genre_ratings.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(list(sorted_genre_ratings.keys()), list(sorted_genre_ratings.values()), color='lightgreen')
        ax.set_xlabel('Average Rating')
        ax.set_title('Top 10 Highest Rated Genres in Your Cluster')
        
        # Add rating labels to bars
        for i, v in enumerate(sorted_genre_ratings.values()):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center')
            
        plt.tight_layout()
        st.pyplot(fig)
    
    # Top rated movies
    st.subheader("Top Rated Movies in Your Cluster")
    
    # Get top rated movies
    top_rated = clustered_df.sort_values('rating', ascending=False).head(2)
    
    # Display movie cards in a row
    cols = st.columns(2)
    for i, (_, movie) in enumerate(top_rated.iterrows()):
        movie_info = get_movie_info(movie['title'])
        if movie_info:
            with cols[i]:
                poster_url = movie_info.get('poster')
                if not poster_url:  # If poster is None, use a placeholder image
                    poster_url = "https://via.placeholder.com/150"
                    st.image(poster_url, width=150)
                st.markdown(f"**{movie_info['name']}**")
                st.markdown(f"Rating: {movie['rating']:.1f}/5")

def display_similar_users(gender, age, occupation, cluster_number):
    df_users = load_users_dataset()
    
    # Check if 'cluster' column exists, if not, add it
    if 'cluster' not in df_users.columns:
        # Load the KMeans model
        kmeans_model = joblib.load("cluster.h5")
        
        # Process each user to predict their cluster
        clusters = []
        for _, user in df_users.iterrows():
            # Convert gender to binary
            gender_binary = 0 if user['gender'] == 'M' else 1
            
            # Get user's age
            user_age = user['age']
            
            # Get user's occupation
            user_occupation = user['occupation']
            
            # Get state from zipcode
            zipcode = str(user['zip-code']).split('-')[0]
            user_state = get_state(zipcode)
            
            # Load state encoder
            state_encoder = joblib.load("state_Encoder.h5")
            try:
                state_encoded = state_encoder.transform([user_state])
                # Predict cluster
                user_data = [[gender_binary, user_age, user_occupation, state_encoded[0]]]
                cluster = kmeans_model.predict(user_data)[0]
            except:
                # Default to the current user's cluster if there's an issue
                cluster = cluster_number
                
            clusters.append(cluster)
        
        # Add cluster column to the dataframe
        df_users['cluster'] = clusters
    
    # Filter users in the same cluster
    cluster_users = df_users[df_users['cluster'] == cluster_number]
    
    if cluster_users.empty:
        st.warning("No similar users found in this cluster.")
        return
    
    # Display demographics of the cluster
    st.subheader("Cluster Demographics")
    
    # Create a layout with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        # Map M/F to Male/Female for display consistency
        cluster_users['display_gender'] = cluster_users['gender'].map({'M': 'Male', 'F': 'Female'})
        gender_counts = cluster_users['display_gender'].value_counts()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
               colors=['lightblue', 'lightpink'], startangle=90)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)
    
    with col2:
        # Age distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(cluster_users['age'], bins=10, kde=True, ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    # Occupation distribution
    st.subheader("Top Occupations in Your Cluster")
    
    # Map numeric occupation codes to names if needed
    if all(isinstance(x, (int, float)) for x in cluster_users['occupation']):
        occupations_list = ["Accountant", "Actor", "Architect", "Artist", "Astronaut", "Athlete", 
                           "Author", "Baker", "Banker", "Barista", "Bartender", "Biologist", 
                           "Butcher", "Carpenter", "Chef", "Chemist", "Civil engineer", "Dentist", 
                           "Doctor", "Electrician", "Engineer", "Firefighter", "Flight attendant", 
                           "Graphic designer", "Hairdresser", "Journalist", "Lawyer", "Librarian", 
                           "Mechanic", "Musician", "Nurse", "Pharmacist", "Photographer", "Pilot", 
                           "Police officer", "Professor", "Programmer", "Psychologist", "Scientist", 
                           "Software developer", "Teacher", "Translator", "Veterinarian", 
                           "Waiter/Waitress", "Web developer", "Writer"]
        
        # Create a mapping function that handles out-of-range indices
        def map_occupation(occ_id):
            try:
                occ_id = int(occ_id)
                if 0 <= occ_id < len(occupations_list):
                    return occupations_list[occ_id]
                else:
                    return f"Occupation {occ_id}"
            except:
                return f"Occupation {occ_id}"
        
        cluster_users['occupation_name'] = cluster_users['occupation'].apply(map_occupation)
        occupation_counts = cluster_users['occupation_name'].value_counts().head(10)
    else:
        occupation_counts = cluster_users['occupation'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(occupation_counts.index, occupation_counts.values, color='lightgreen')
    ax.set_xlabel('Count')
    ax.set_title('Top Occupations')
    
    # Add count labels to bars
    for i, v in enumerate(occupation_counts.values):
        ax.text(v + 0.1, i, str(v), va='center')
        
    plt.tight_layout()
    st.pyplot(fig)

def display_recommendations():
    if 'user_cluster' not in st.session_state:
        st.warning("Please complete your profile analysis first to get recommendations.")
        return
    
    cluster_number = st.session_state['user_cluster']
    
    st.title("🍿 Movie Recommendations")
    st.subheader(f"Based on Cluster #{cluster_number}")
    
    # Load movie data
    df_movies = load_movies_dataset()
    
    # Filter movies for the cluster
    clustered_df = df_movies[df_movies['cluster'] == cluster_number]
    
    if clustered_df.empty:
        st.warning("No recommendations available for this cluster.")
        return
    
    # Create tabs for different recommendation types
    tab1, tab2 = st.tabs(["Popular in Your Cluster", "Highest Rated in Your Cluster"])
    
    with tab1:
        # Get the top movies by frequency
        top_movies = clustered_df['title'].value_counts().head(10)
        
        st.subheader("Most Popular Movies in Your Cluster")
        
        # Create rows with 2 movies each
        for i in range(0, min(6, len(top_movies)), 2):
            cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(top_movies):
                    movie_title = top_movies.index[i + j]
                    movie_count = top_movies.iloc[i + j]
                    
                    movie_info = get_movie_info(movie_title)
                    if movie_info:
                        display_movie_card(movie_info, cols[j], show_plot=False)
                        cols[j].caption(f"Watched by {movie_count} users in your cluster")
    
    with tab2:
        # Get top rated movies with at least 5 ratings
        movie_ratings = clustered_df.groupby('title').agg({
            'rating': ['mean', 'count']
        })
        movie_ratings.columns = ['avg_rating', 'count']
        top_rated = movie_ratings[movie_ratings['count'] >= 5].sort_values('avg_rating', ascending=False).head(10)
        
        st.subheader("Highest Rated Movies in Your Cluster")
        
        # Create rows with 2 movies each
        for i in range(0, min(6, len(top_rated)), 2):
            cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(top_rated):
                    movie_title = top_rated.index[i + j]
                    avg_rating = top_rated.iloc[i + j]['avg_rating']
                    count = top_rated.iloc[i + j]['count']
                    
                    movie_info = get_movie_info(movie_title)
                    if movie_info:
                        display_movie_card(movie_info, cols[j], show_plot=False)
                        cols[j].caption(f"Average Rating: {avg_rating:.2f}/5 (from {count} ratings)")

def content_based_recommendations():
    st.title("🧩 Content-Based Recommendations")
    
    # Load movie data
    df_movies = load_movies_for_content()
    
    # Sort movie titles alphabetically for easier selection
    sorted_movies = sorted(df_movies['title'].unique())
    
    # Movie selection with search functionality
    favorite_movie = st.selectbox(
        "Select Your Favorite Movie", 
        sorted_movies,
        index=0
    )
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            # Get similar movies
            similar_movies = get_similar_movies(favorite_movie, top_n=6)
            
            if similar_movies.empty:
                st.error("Could not find similar movies. Please try another movie.")
                return
            
            # Display favorite movie info
            st.subheader("Your Selected Movie")
            favorite_movie_info = get_movie_info(favorite_movie)
            
            if favorite_movie_info:
                col = st.columns(1)[0]
                display_movie_card(favorite_movie_info, col)
            
            # Display similar movies
            st.subheader("Similar Movies Based on Genre")
            
            # Create rows with 3 movies each
            for i in range(0, len(similar_movies), 3):
                cols = st.columns(3)
                
                for j in range(3):
                    if i + j < len(similar_movies):
                        movie = similar_movies.iloc[i + j]
                        movie_info = get_movie_info(movie['title'])
                        
                        if movie_info:
                            with cols[j]:
                                poster_url = movie_info.get('poster')
                                if not poster_url:  # If poster is None, use a placeholder image
                                poster_url = "https://via.placeholder.com/150"
                                st.image(poster_url, width=200)
                                st.markdown(f"**{movie_info['name']}**")
                                st.caption(f"Genres: {movie['genres']}")
                                st.progress(float(movie['similarity_score']))
                                st.caption(f"Similarity: {movie['similarity_score']:.2f}")

def item_based_recommendations():
    st.title("🔄 Item-Based Collaborative Filtering")
    
    # Load movie data
    df_movies = load_movies_for_content()
    
    # Sort movie titles alphabetically for easier selection
    sorted_movies = sorted(df_movies['title'].unique())
    
    # Movie selection with search functionality
    favorite_movie = st.selectbox(
        "Select Your Favorite Movie", 
        sorted_movies,
        index=0
    )
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies based on user ratings..."):
            # Get similar movies
            similar_movies = get_item_based_recommendations(favorite_movie, top_n=6)
            
            if similar_movies.empty:
                st.error("Could not find similar movies. Please try another movie.")
                return
            
            # Display favorite movie info
            st.subheader("Your Selected Movie")
            favorite_movie_info = get_movie_info(favorite_movie)
            
            if favorite_movie_info:
                col = st.columns(1)[0]
                display_movie_card(favorite_movie_info, col)
            
            # Display similar movies
            st.subheader("Movies Liked by People Who Like This Movie")
            
            # Create rows with 3 movies each
            for i in range(0, len(similar_movies), 3):
                cols = st.columns(3)
                
                for j in range(3):
                    if i + j < len(similar_movies):
                        movie = similar_movies.iloc[i + j]
                        movie_info = get_movie_info(movie['title'])
                        
                        if movie_info:
                            with cols[j]:
                                poster_url = movie_info.get('poster')
                                if not poster_url:  # If poster is None, use a placeholder image
                                    poster_url = "https://via.placeholder.com/150"
                                    st.image(poster_url, width=200)
                                st.markdown(f"**{movie_info['name']}**")
                                st.caption(f"Genres: {movie['genres']}")
                                st.progress(float(movie['similarity_score']))
                                st.caption(f"Similarity: {movie['similarity_score']:.2f}")

def hybrid_recommendations():
    st.title("🔀 Hybrid Recommendation System")
    
    if 'user_cluster' not in st.session_state:
        st.warning("Please complete your profile analysis first to use the hybrid recommendation system.")
        return
    
    # Get user information
    cluster_number = st.session_state['user_cluster']
    
    # Load movie data
    df_movies = load_movies_dataset()
    df_content_movies = load_movies_for_content()
    
    # Get top movies in user's cluster
    clustered_df = df_movies[df_movies['cluster'] == cluster_number]
    top_cluster_movies = clustered_df['title'].value_counts().head(20).index.tolist()
    
    # Let user select a movie from their cluster's favorites
    st.subheader("Step 1: Select a movie you like from your cluster's favorites")
    selected_movie = st.selectbox("Select a Movie", top_cluster_movies)
    
    if st.button("Generate Hybrid Recommendations"):
        with st.spinner("Creating personalized recommendations..."):
            # Get content-based recommendations
            content_recs = get_similar_movies(selected_movie, top_n=10)
            
            # Get collaborative filtering recommendations
            collab_recs = get_item_based_recommendations(selected_movie, top_n=10)
            
            # Combine recommendations
            combined_recs = pd.concat([
                content_recs.assign(source='content'),
                collab_recs.assign(source='collaborative')
            ])
            
            # Remove duplicates, keeping the higher similarity score
            combined_recs = combined_recs.sort_values('similarity_score', ascending=False)
            combined_recs = combined_recs.drop_duplicates(subset=['title'])
            
            # Get top 3 recommendations
            top_recs = combined_recs.head(3)
            
            # Display selected movie info
            st.subheader("Your Selected Movie")
            selected_movie_info = get_movie_info(selected_movie)
            
            if selected_movie_info:
                col = st.columns(1)[0]
                display_movie_card(selected_movie_info, col)
            
            # Display hybrid recommendations
            st.subheader("Your Personalized Recommendations")
            
            # Create rows with 3 movies each
            for i in range(0, len(top_recs), 3):
                cols = st.columns(3)
                
                for j in range(3):
                    if i + j < len(top_recs):
                        movie = top_recs.iloc[i + j]
                        movie_info = get_movie_info(movie['title'])
                        
                        if movie_info:
                            with cols[j]:
                                poster_url = movie_info.get('poster')
                                if not poster_url:  # If poster is None, use a placeholder image
                                    poster_url = "https://via.placeholder.com/150"
                                    st.image(poster_url, width=200)
                                st.markdown(f"**{movie_info['name']}**")
                                source_label = "Content-Based" if movie['source'] == 'content' else "Collaborative Filtering"
                                st.caption(f"Source: {source_label}")
                                st.progress(float(movie['similarity_score']))
                                st.caption(f"Similarity: {movie['similarity_score']:.2f}")

def about():
    st.title("ℹ️ About This Recommendation System")
    
    st.markdown("""
    ## Movie Recommendation System
    
    This application uses multiple recommendation techniques to suggest movies:
    
    ### 1. User Clustering
    - Groups users with similar demographics and preferences
    - Provides recommendations based on what similar users enjoy
    
    ### 2. Content-Based Filtering
    - Recommends movies based on genre similarity
    - Finds movies with similar content to your favorites
    
    ### 3. Item-Based Collaborative Filtering
    - Analyzes user rating patterns
    - Suggests movies that users with similar tastes enjoyed
    
    ### 4. Hybrid Recommendations
    - Combines multiple recommendation techniques
    - Produces more diverse and personalized suggestions
    
    ### Data Sources
    This system uses several datasets:
    - Movie metadata (titles, genres)
    - User ratings
    - User demographic information
    
    ### Technologies Used
    - Streamlit for the web interface
    - Scikit-learn for machine learning algorithms
    - IMDb API for movie information and posters
    - Pandas for data manipulation
    - Matplotlib and Seaborn for visualizations
    """)

PAGES = {
    "User Profile & Clustering": main,
    "Movie Recommendations": display_recommendations,
    "Content-Based Recommendations": content_based_recommendations,
    "Item-Based Recommendations": item_based_recommendations,
    "Hybrid Recommendations": hybrid_recommendations,
    "About": about
}

def run_app():
    # Add custom CSS
    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎬 Movie Recommender")
    
    # Add user profile info to sidebar if available
    if 'user_profile' in st.session_state:
        profile = st.session_state['user_profile']
        cluster = st.session_state.get('user_cluster', 'Unknown')
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Your Profile")
        st.sidebar.markdown(f"**Gender:** {profile['gender']}")
        st.sidebar.markdown(f"**Age:** {profile['age']}")
        st.sidebar.markdown(f"**Occupation:** {profile['occupation']}")
        st.sidebar.markdown(f"**Cluster:** {cluster}")
        st.sidebar.markdown("---")
    
    # Navigation options
    page = st.sidebar.radio("Navigation", list(PAGES.keys()))
    
    # Run the selected page
    PAGES[page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 Movie Recommendation System")

if __name__ == "__main__":
    run_app()
