import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image, display
from bs4 import BeautifulSoup
import ast
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
df = pd.read_csv(r"C:\Users\user\Desktop\Final File for project 2 1.1.2025\steam.csv")
#df=df.head(100)
pd.set_option('display.max_columns', None)
#df.columns

def clean_conc(row): 
    genres = str(row["genres"]).replace(";", " ")
    steamspy_tags = str(row["steamspy_tags"]).replace(' ', '').replace(";", " ")
    categories = str(row["categories"]).replace(' ', '').replace(";", " ").replace("-", "")
    concat = genres + " " + categories + ' ' + str(row['publisher']) + ' ' + steamspy_tags
    return concat

# Prepare your data (df represents your DataFrame)
df_selected = df[['appid', 'name', 'publisher', 'categories', 'genres', 'positive_ratings', 'negative_ratings', 'steamspy_tags', 'price', 'average_playtime', 'owners']]
df_selected = df_selected.fillna("Unknown").astype(str)
df_selected["tfidf_input"] = df_selected.apply(clean_conc, axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_selected['tfidf_input'])

def get_recommendations(game_id, top_n=10):
    try:
        # Check if the game exists in the dataset
        if game_id in df_selected['appid'].values:
            # Get the index of the game that matches the ID
            game_index = df_selected[df_selected['appid'] == game_id].index[0]

            # Get the pairwise similarity scores of all games with that game
            sim_scores = cosine_similarity(tfidf_matrix[game_index], tfidf_matrix).flatten()

            # Sort the games based on the similarity scores (excluding the current game)
            top_indices = sim_scores.argsort()[-(top_n + 1):-1][::-1]

            # Get the top N most similar games
            top_similar_games = df_selected.iloc[top_indices][['name', 'appid', 'price']]

            return top_similar_games  # Ensure this line is added to return the top games

        else:
            print(f"Game with ID {game_id} does not exist in the dataset.")
            return None  # Ensure None is returned if the game doesn't exist in the dataset

    except IndexError:
        print(f"The game with ID {game_id} does not exist in the dataset.")
        return None  # Ensure None is returned on error


def get_app_id(game_name):
    # Make sure the game name is URL encoded to handle special characters
    game_name_encoded = game_name.replace(' ', '+')  # Replaces spaces with '+' to match URL encoding
    
    # Make the request with the exact game name
    response = requests.get(url=f'https://store.steampowered.com/search/?term={game_name_encoded}&category1=998', headers={'User-Agent': 'Mozilla/5.0'})
    
    # Parse the response with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Try to find the app id in the HTML
    try:
        app_id = soup.find(class_='search_result_row')['data-ds-appid']
        return app_id
    except (AttributeError, TypeError):
        return None 

def get_game_data(game_id):
    
    steam_game_data = get_single_steam_game_data(game_id)
    steamspy_game_data = get_single_steamspy_game_data(game_id)
    steam_game_data.rename(columns={'steam_appid': 'appid'}, inplace=True)
    all_data = pd.merge(steam_game_data, steamspy_game_data, on='appid')
    all_data=process_categories_and_genres(all_data)
    all_data=all_data.fillna("Unknown")
    all_data["category+genre+publisher"]=df_selected.apply(clean_conc, axis=1)
    return all_data
def process_categories_and_genres(df):
    df = df.copy()
    df = df[(df['categories'].notnull()) & (df['genres'].notnull())]
    
    for col in ['categories', 'genres']:
        df[col] = df[col].apply(lambda x: ';'.join(item['description'] for item in x))
    
    return df   
def get_request(url, parameters=None):
    """Make an HTTP GET request to the specified URL with optional parameters."""
    try:
        response = requests.get(url, params=parameters)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print("Error during HTTP request:", e)
        return None    

# Function to get data for a single game from Steam Store API
def get_single_steam_game_data(appid):
    """Retrieve data for a single game from Steam Store API."""
    url = "http://store.steampowered.com/api/appdetails/"
    parameters = {"appids": appid}
    
    json_data = get_request(url, parameters=parameters)
    json_app_data = json_data[str(appid)]
    
    if json_app_data['success']:
        data = json_app_data['data']
        # Convert the data to a DataFrame
        game_data = pd.DataFrame([data])
    else:
        # If the request is not successful, return an empty DataFrame
        game_data = pd.DataFrame()
        
    return game_data
def  get_single_steamspy_game_data (appid):
    """Function to retrieve data of a single game from the SteamSpy API."""
    steamspy_data = parse_steamspy_request(appid)
    # Transpose the DataFrame
    steamspy_data = steamspy_data.T.reset_index()
    return steamspy_data

def parse_steamspy_request(appid):
    """Function to retrieve data of a single game from the SteamSpy API."""
    url = "https://steamspy.com/api.php"
    parameters = {"request": "appdetails", "appid": appid}
    
    json_data = requests.get(url, parameters).json()
    return pd.DataFrame.from_dict(json_data, orient='index')


def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        print(url+appid)
        return response.json()

def get_n_reviews(appid, n=100):


    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 9223372036854775807,
            'review_type' : 'all',
            'purchase_type' : 'all'
            }

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        if len(response['reviews']) < 100: break

    return pd.DataFrame(reviews)

def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()

def stats(reviews, appid):
    # Ensure average_playtime is numeric
    df_selected['average_playtime'] = pd.to_numeric(df_selected['average_playtime'], errors='coerce')
    
    # Handle NaN values
    df_selected['average_playtime'].fillna(0, inplace=True)
    
    # Calculate the average game time in hours
    avr_t_g = df_selected[df_selected['appid'] == appid]['average_playtime']
    
    # Safely extract 'playtime_at_review' and 'playtime_forever' from the 'author' column
    reviews['playtime_at_review'] = reviews['author'].apply(lambda x: x.get('playtime_at_review', 0))
    reviews['playtime_forever'] = reviews['author'].apply(lambda x: x.get('playtime_forever', 0))
    
    averge_time_r_postive = reviews[reviews['voted_up'] == True]['playtime_at_review'].mean() / 60
    averge_time_r_negative = reviews[reviews['voted_up'] == False]['playtime_at_review'].mean() / 60
    averge_time_r = reviews['playtime_at_review'].mean() / 60
    averge_time_p = df_selected[df_selected['appid'] == appid]['average_playtime'].mean() / 60

    # Prepare data for the table
    data = {
        'Metric': [
            'Avg hours when positive review left (hours):',
            'Avg hours when negative review left (hours):',
            'Avg hours played by reviewers (hours):',
            'Avg total hours played (hours): '
        ],
        'Value': [
            f'{averge_time_r_postive:.0f}',
            f'{averge_time_r_negative:.0f}',
            f'{averge_time_r:.0f}',
            f'{averge_time_p:.0f}'
        ]
    }

    # Convert the data to a DataFrame
    df_stats = pd.DataFrame(data)

    # Display the table in Streamlit
    st.dataframe(df_stats,height=178,width=700)

def pie_chart(game_data):
    positive_reviews = game_data['positive'][0]
    negative_reviews = game_data['negative'][0]

    # Data for the pie chart
    labels = ['Positive Reviews', 'Negative Reviews']
    sizes = [positive_reviews, negative_reviews]
    colors = ['#66c2a5', '#fc8d62']  

    # Create a larger figure to accommodate the pie chart
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust size as needed
    
    # Creating a 2D pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=200, 
                                      pctdistance=0.6, labeldistance=0.5, explode=(0.1, 0), shadow=True)

    # Adjust the position of the text labels and autotexts (percentages)
    for text in texts:
        text.set_y(text.get_position()[1] - 0.065)  # Adjust the label text downward

    for autotext in autotexts:
        autotext.set_y(autotext.get_position()[1] - 0.0075)  # Adjust the percentage text downward

    # Title and customization
    ax.set_title('Ratio of Positive and Negative Reviews', pad=20)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Tighten layout to minimize white space
    plt.tight_layout()

    # Display the pie chart in Streamlit
    st.pyplot(fig)  # Render the Matplotlib figure in Streamlit

    # Clear the plot to avoid overlap if the function is called multiple times
    plt.clf()


def wrod_cloud(reviews):

    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Initialize an empty string to store reviews
    rev = ""

    for line in reviews['review']:
        rev += str(line)

    # Tokenize the reviews
    tokens = word_tokenize(rev)

    # Remove stopwords, punctuation, and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # POS tagging
    tagged_tokens = pos_tag(tokens)

    # Filter out adjectives
    adjectives = [word for word, pos in tagged_tokens if pos.startswith('JJ')]

    # Count the most common 25 words
    common_words = Counter(adjectives).most_common(100)

    # Generate word cloud
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(dict(common_words))

    # Plot the word cloud
    plt.figure(figsize=(4, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    st.pyplot(plt)  # Render the word cloud in Streamlit

    # Clear the plot to avoid overlap if the function is called multiple times
    plt.clf()

def semantic_by_topic(reviews):
    lemmatizer = WordNetLemmatizer()
    
    # Text preprocessing
    reviews['processed_text'] = reviews['review'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x.lower()) if word.isalpha() and word not in stopwords.words('english')])
    )

    # Sentiment analysis
    reviews['sentiment'] = reviews['processed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    topics_of_interest = {
        'gameplay': ["Mechanics", "Controls", "Immersion", "Dynamics", "Interaction", "Challenge", "Progression", "Difficulty", "Replayability", "Depth", "Balance", "Pace", "Strategy", "Tactics", "Exploration",
                     "Combat", "Puzzle-solving", "Skill", "Mastery", "Engagement", 'multiplayer', 'gameplay', 'movement', 'boring', 'exciting', 'accuracy', 'accurate'], 
        'story': ["Plot", "Narrative", "Characters", "Dialogue", "Storytelling", "Lore", "Backstory", "Setting", "Protagonist", "Antagonist", "Conflict", "Resolution", "Twists", "Turns", "Quests", "Missions", "World-building", "Cutscenes", "Choices", 'expected', 'unexpected'],
        'performance': ["FPS", "smoothness", "optimization", "Responsiveness", "Loading times", "Rendering", "Graphics", "quality", "Frame", "delay", "performance", "Network", "latency", "Bandwidth", "stability", "Client-server", "hardware", 'performance', 'lag', 'slow', 'crash', 'server', 'ping']
    }

    sentiment_data = {}
    for topic, keywords in topics_of_interest.items():
        review_topic = reviews[reviews['processed_text'].apply(lambda x: any(keyword in x for keyword in keywords))]
        positive_sentiment = review_topic[review_topic['sentiment'] > 0]['sentiment'].count()
        negative_sentiment = review_topic[review_topic['sentiment'] < 0]['sentiment'].count()
        sentiment_data[topic] = {'Positive': positive_sentiment, 'Negative': -negative_sentiment}

    # Create DataFrame for sentiment data
    df_sentiment = pd.DataFrame(sentiment_data)

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Positive': '#66c2a5', 'Negative': '#fc8d62'}
    df_sentiment.transpose().plot(kind='barh', ax=ax, color=[colors['Positive'], colors['Negative']], edgecolor='black')

    # Set the title and labels, increase their font sizes
    ax.set_title('Sentiment Analysis by Topic', fontsize=20)
    ax.set_xlabel('Sentiment', fontsize=15)
    ax.set_ylabel('Topics', fontsize=15)

    # Increase the font size of the ticks
    ax.tick_params(axis='both', labelsize=12)
    
    ax.legend()
    ax.grid()

    ax.set_facecolor('white')  # Set the background color to white

    # Adjust the position of the negative bars to align them with positive bars
    for bar in ax.patches:
        if bar.get_width() < 0:  # Negative bars
            bar.set_y(bar.get_y() - 0.25)  # Adjust y-position to move the bar down further

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)  # Render the Matplotlib figure in Streamlit

    # Clear the plot to avoid overlap if the function is called multiple times
    plt.clf()

def the_whole_proces():
    game_name = input("Game name")
    game_id=get_app_id(game_name)
    game_data = get_game_data(game_id)
    get_recommendations(game_id)

    revewies=get_n_reviews(f'{game_id}')
    
    stats(revewies,game_id)
    pie_chart(game_data)
    wrod_cloud(revewies)
    semantic_by_topic(revewies)



# %%
#the_whole_proces()


# %%
# ''' Frontend '''

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from wordcloud import WordCloud
from textblob import TextBlob

# Define the Streamlit App header
st.write(""" <h2> <b style="color:red"> Steam-Pulse</b> </h2>""", unsafe_allow_html=True)
st.write("###")
st.write(""" <p> Welcome to <b style="color:red">Steam-Pulse</b> this free games recommendation engine suggests games based on your interest </p>""", unsafe_allow_html=True)
st.write("##")

# Input field for game name
game_name = st.text_input("Enter the name of the game you'd like recommendations for:")


if st.button("Recommend"):
    if game_name.strip() == "":
        st.error("Please enter a valid game name.")
    else:
        try:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Recommendations", "Game Statistics", "Review Analysis", "Word Cloud", "Sentiment Analysis"])
            # Get the game ID and data
            game_id = get_app_id(game_name)
            if game_id is None:
                st.error(f"Game '{game_name}' not found on Steam.")
            else:
                game_data = get_game_data(game_id)
                reviews = get_n_reviews(f"{game_id}")

                # Tab 1: Recommendations
                with tab1:
            
                    st.write(f"Top recommendations for **{game_name}**:")
                    recommended_games = get_recommendations(game_id)
                
                    if recommended_games is not None:
                        for i, (name, appid, price) in enumerate(zip(recommended_games['name'], recommended_games['appid'], recommended_games['price']), start=1):
                            image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
                            st.write(f"{i}. [{name}](https://store.steampowered.com/app/{appid})")
                            st.image(image_url)
                    else:
                        st.error(f"No recommendations available for {game_name}.")

                # Tab 2: Game Statistics
                with tab2:
                    st.write("### Game Statistics")
                    stats(reviews, game_id)

                # Tab 3: Pie Chart
                with tab3:
                    st.write("### Ratio of Positive and Negative Reviews")
                    pie_chart(game_data)

                # Tab 4: Word Cloud
                with tab4:
                    st.write("### Word Cloud of Reviews")
                    wrod_cloud(reviews)

                # Tab 5: Sentiment Analysis
                with tab5:
                    st.write("### Sentiment Analysis by Topic")
                    semantic_by_topic(reviews)

        except Exception as e:
            st.error(f"An error occurred: {e}")


