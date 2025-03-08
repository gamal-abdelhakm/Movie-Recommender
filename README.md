# Movie Recommendation System
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://share.streamlit.io/gamal-abdelhakm/Movie-Recommender/main/app.py)

This repository implements a movie recommendation system using various techniques such as content-based filtering, item-based collaborative filtering, and machine learning.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Sources](#data-sources)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/gamal-abdelhakm/Movie-Recommender.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Movie-Recommender
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On MacOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can access the deployed application using the following link:
[Movie Recommendation System](https://your-deployment-link.com)

Alternatively, you can run the application locally:

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to:
    ```
    http://localhost:8501
    ```

## Features

### User Profile & Clustering
- Enter your demographic information to find out which user cluster you belong to.
- View cluster demographics and top movies in your cluster.

### Movie Recommendations
- Get movie recommendations based on your user profile cluster.

### Content-Based Recommendations
- Receive recommendations based on the genres of your favorite movies.

### Item-Based Collaborative Filtering
- Get recommendations based on user rating patterns and similarities.

### Hybrid Recommendations
- Combines content-based and collaborative filtering for more diverse and personalized suggestions.

### About
- Learn more about the recommendation techniques and data sources used in this system.

## Data Sources

This system uses several datasets:
- **Movies**: Metadata such as titles, genres, and more.
- **Ratings**: User ratings of movies.
- **Users**: Demographic information of users.

## Technologies Used

- **Streamlit**: Web interface framework.
- **Scikit-learn**: Machine learning algorithms.
- **IMDb API**: Movie information and posters.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualizations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features.

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/my-new-feature
    ```
3. Commit your changes:
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature/my-new-feature
    ```
5. Open a pull request.
