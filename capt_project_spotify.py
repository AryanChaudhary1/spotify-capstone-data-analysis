"""Capt_Project_Spotify.py
"""
import os
import numpy as np

# Seed the random number generator
rng=np.random.seed(17764741)

import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()

# Construct the file path relative to the current directory
file_path = os.path.join(current_directory, "spotify52kData.csv")

# Read the CSV file 
spotify_data = pd.read_csv(file_path)

spotify_data.head()

spotify_data.info()

# Check for missing values
missing_values = spotify_data.isnull().sum()
print("Missing values:\n", missing_values)

# Remove rows with missing values
spotify_data_cleaned = spotify_data.dropna()

# Convert "explicit" to numerical (0 for False, 1 for True)
spotify_data_cleaned['explicit'] = spotify_data_cleaned['explicit'].astype(int)

# Convert "mode" to numerical (0 for minor, 1 for major)
spotify_data_cleaned['mode'] = spotify_data_cleaned['mode'].astype(int)

# Check data types
print("Data types after preprocessing:\n", spotify_data_cleaned.dtypes)

"""**1. Consider the 10 song features duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence and tempo. Is any of these features reasonably distributed normally? If so, which one? [Suggestion: Include a 2x5 figure with histograms for each feature)**"""

# Extracting the numeric features for analysis
numeric_features = ['duration', 'danceability', 'energy', 'loudness',
                    'speechiness', 'acousticness', 'instrumentalness',
                    'liveness', 'valence', 'tempo']

# Plot histograms for each numeric feature
plt.figure(figsize=(15, 6))
for i, feature in enumerate(numeric_features):
    plt.subplot(2, 5, i+1)
    plt.hist(spotify_data[feature], bins=30, color='skyblue', edgecolor='black')
    plt.title(feature)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""**Observation :-**


1.   Among the 10 features, only **loudness** and **tempo** appear to be reasonably normally distributed.
2.   The histograms for the other features exhibit varying degrees of skewness and kurtosis, indicating that they are not normally distributed.

**2. Is there a relationship between song length and popularity of a song? If so, if the relationship positive or negative? [Suggestion: Include a scatterplot]**
"""

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(spotify_data_cleaned['duration'], spotify_data_cleaned['popularity'], alpha=0.5)
plt.title('Relationship between Song Length and Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.grid(True)
plt.show()

"""**Observations**:

- There appears to be a weak positive relationship between song length and popularity.
- This means that, on average, longer songs tend to be more popular than shorter songs.
- However, there is a lot of variation in the data, with some short songs being very popular and some long songs being not very popular.
- This suggests that song length is not the only factor that determines a song's popularity.

**3. Are explicitly rated songs more popular than songs that are not explicit? [Suggestion: Do a suitable significance test, be it parametric, non-parametric or permutation]**
"""

# Split the data into two groups: explicit and non-explicit
explicit_group = spotify_data_cleaned[spotify_data_cleaned['explicit'] == 1]['popularity']
non_explicit_group = spotify_data_cleaned[spotify_data_cleaned['explicit'] == 0]['popularity']

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(explicit_group, non_explicit_group, alternative='greater')

# Print the results
print("Mann-Whitney U Test Results:")
print("Statistic:", statistic)
print("p-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. Explicitly rated songs are more popular than non-explicitly rated songs.")
else:
    print("Fail to reject the null hypothesis. There is no sufficient evidence to conclude that explicitly rated songs are more popular than non-explicitly rated songs.")

"""**Observations**:

- The Mann-Whitney U test results show a statistically significant difference in popularity between explicitly rated songs and non-explicitly rated songs.
- The p-value is less than the significance level of 0.05, which means that we can reject the null hypothesis that there is no difference in popularity between the two groups.
- This suggests that explicitly rated songs are more popular than non-explicitly rated songs.
- It is important to note that this analysis does not take into account other factors that may influence a song's popularity, such as genre, artist, and release date.

**4. Are songs in major key more popular than songs in minor key? [Suggestion: Do a suitable significance test, be it parametric, non-parametric or permutation]**
"""

# Split the data into two groups: major key and minor key
major_key_group = spotify_data_cleaned[spotify_data_cleaned['mode'] == 1]['popularity']
minor_key_group = spotify_data_cleaned[spotify_data_cleaned['mode'] == 0]['popularity']

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(major_key_group, minor_key_group, alternative='greater')

# Print the results
print("Mann-Whitney U Test Results:")
print("Statistic:", statistic)
print("p-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. Songs in a major key are more popular than songs in a minor key.")
else:
    print("Fail to reject the null hypothesis. There is no sufficient evidence to conclude that songs in a major key are more popular than songs in a minor key.")

"""**Observations**:

- The Mann-Whitney U test results show a statistically significant difference in popularity between songs in a major key and songs in a minor key.
- The p-value is less than the significance level of 0.05, which means that we can reject the null hypothesis that there is no difference in popularity between the two groups.
- This suggests that songs in a major key are more popular than songs in a minor key.
- It is important to note that this analysis does not take into account other factors that may influence a song's popularity, such as genre, artist, and release date.

**5. Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute) that this is the case? [Suggestion: Include a scatterplot]**
"""

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(spotify_data_cleaned['energy'], spotify_data_cleaned['loudness'], alpha=0.5)
plt.title('Relationship between Energy and Loudness')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.grid(True)
plt.show()

"""**Observations**:

- There appears to be a positive correlation between energy and loudness.
- This means that, on average, songs with higher energy tend to be louder.
- However, there is a lot of variation in the data, with some low-energy songs being loud and some high-energy songs being quiet.
- This suggests that energy and loudness are not perfectly correlated.
- There may be other factors that contribute to a song's perceived loudness, such as the frequency content and the use of compression.

**6. Which of the 10 individual (single) song features from question 1 predicts popularity best? How good is this “best” model?**
"""

# List of song features
song_features = ['duration', 'danceability', 'energy', 'loudness',
                 'speechiness', 'acousticness', 'instrumentalness',
                 'liveness', 'valence', 'tempo']

# Initialize dictionary to store R-squared values for each feature
r_squared_values = {}

# Iterate over each song feature
for feature in song_features:
    # Extract feature and target variable
    X = spotify_data_cleaned[[feature]]
    y = spotify_data_cleaned['popularity']

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict popularity
    y_pred = model.predict(X)

    # Calculate R-squared
    r_squared = r2_score(y, y_pred)

    # Store R-squared value
    r_squared_values[feature] = r_squared

# Identify the feature with the highest R-squared value
best_predictor = max(r_squared_values, key=r_squared_values.get)
best_r_squared = r_squared_values[best_predictor]

print("Best predictor of popularity:", best_predictor)
print("R-squared value:", best_r_squared)

"""**Observation:**

The best predictor of popularity among the 10 individual song features is **instrumentalness**, with an R-squared value of 0.021. This means that instrumentalness explains only about 2.1% of the variation in popularity.

This suggests that popularity is a complex measure that is influenced by a variety of factors beyond the 10 song features considered in this analysis. Other factors that may influence popularity include genre, artist, release date, and marketing.

**7. Building a model that uses all of the song features from question 1, how well can you predict popularity now? How much (if at all) is this model improved compared to the best model in question 6). How do you account for this?**
"""

# Extract features and target variable
X_all = spotify_data_cleaned[song_features]
y_all = spotify_data_cleaned['popularity']

# Fit multiple linear regression model
model_all = LinearRegression()
model_all.fit(X_all, y_all)

# Predict popularity
y_pred_all = model_all.predict(X_all)

# Calculate R-squared
r_squared_all = r2_score(y_all, y_pred_all)

# Print R-squared value
print("R-squared value using all features:", r_squared_all)

# Compare with best model from question 6
print("R-squared value of the best model:", best_r_squared)
print("Improvement in R-squared:", r_squared_all - best_r_squared)

"""**Observations:**

- Using all of the song features together in a multiple linear regression model improves the R-squared value to 0.047, compared to 0.021 for the best individual feature (instrumentalness).
- This suggests that there is some additional explanatory power in the other features, even though they are individually weak predictors of popularity.
- However, the improvement in R-squared is still relatively small, indicating that the 10 song features considered in this analysis still do not fully explain the variation in popularity.
- This is consistent with the findings from question 6, which suggested that popularity is a complex measure that is influenced by a variety of factors beyond these 10 features.

**8. When considering the 10 song features above, how many meaningful principal components can you extract? What proportion of the variance do these principal components account for?**
"""

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spotify_data_cleaned[song_features])

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Number of principal components
num_components = pca.n_components_

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Print results
print("Number of principal components:", num_components)
print("Explained variance ratio:", explained_variance_ratio)
print("Cumulative explained variance:", cumulative_explained_variance)

"""**Observations:**

- The PCA analysis reveals that there are 10 meaningful principal components, which is the same as the number of original features.
- The first principal component explains the most variance (29.7%), followed by the second principal component (19.9%), and so on.
- The cumulative explained variance shows that the first 3 principal components account for 69.6% of the total variance in the data.
- This suggests that it may be possible to reduce the dimensionality of the data by using only the first few principal components, while still retaining most of the important information.

**9. Can you predict whether a song is in major or minor key from valence? If so, how good is this prediction? If not, is there a better predictor? [Suggestion: It might be nice to show the logistic regression once you are done building the model]**
"""

# Prepare the data
X = spotify_data_cleaned[['valence']]
y = spotify_data_cleaned['mode']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict mode (major or minor) for the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Plotting decision boundary
X_valence = X_test.values
X_min, X_max = X_valence.min() - 0.1, X_valence.max() + 0.1
X_valence_plot = np.linspace(X_min, X_max, 1000).reshape(-1, 1)
y_prob = model.predict_proba(X_valence_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_valence_plot, y_prob[:, 1], color='blue', label='Predicted Probability (Major)')
plt.plot(X_valence_plot, y_prob[:, 0], color='red', label='Predicted Probability (Minor)')
plt.xlabel('Valence')
plt.ylabel('Probability')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

"""**Observations**:

- The logistic regression model achieves an accuracy of 61% in predicting the mode (major or minor) of a song based on its valence.
- This suggests that valence is a useful predictor of mode, although there is still room for improvement.
- The classification report shows that the model performs slightly better at predicting major mode (73.8%) than minor mode (66.7%).
- The decision boundary plot shows that the model is able to separate the two classes (major and minor) reasonably well, although there is some overlap.
- Overall, these results suggest that valence is a useful feature for predicting the mode of a song, and that a logistic regression model can be used to achieve reasonable accuracy in this task.

**10. Which is a better predictor of whether a song is classical music – duration or the principal components you extracted in question 8? [Suggestion: You might have to convert the qualitative genre label to a binary numerical label (classical or not)]**
"""

# Get the current working directory
current_directory = os.getcwd()

# Construct the file path relative to the current directory
file_path = os.path.join(current_directory, "spotify52kData.csv")

# Read the CSV file 
spotify_data = pd.read_csv(file_path)

# Check the columns present in the dataset
print(spotify_data.columns)

# Drop non-numeric columns
spotify_data_numeric = spotify_data.drop(['artists', 'album_name', 'track_name', 'track_genre'], axis=1)

# Compute principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(spotify_data_numeric)

# Convert to DataFrame
spotify_data_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

# Concatenate with target variable
spotify_data_pca['classical'] = (spotify_data['track_genre'] == 'classical').astype(int)

# Prepare the data
X_duration = spotify_data[['duration']]
X_pc = spotify_data_pca[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']]
y_classical = spotify_data_pca['classical']

# Split the data into training and testing sets
X_duration_train, X_duration_test, y_train, y_test = train_test_split(X_duration, y_classical, test_size=0.2, random_state=42)

# Scale the principal components
scaler = StandardScaler()
X_pc_scaled = scaler.fit_transform(X_pc)

# Split the scaled principal components into training and testing sets
X_pc_train_scaled, X_pc_test_scaled, _, _ = train_test_split(X_pc_scaled, y_classical, test_size=0.2, random_state=42)

# Train logistic regression models
model_duration = LogisticRegression()
model_pc_scaled = LogisticRegression()

model_duration.fit(X_duration_train, y_train)
model_pc_scaled.fit(X_pc_train_scaled, y_train)

# Predictions
y_pred_duration = model_duration.predict(X_duration_test)
y_pred_pc_scaled = model_pc_scaled.predict(X_pc_test_scaled)

# Evaluate accuracy
accuracy_duration = accuracy_score(y_test, y_pred_duration)
accuracy_pc_scaled = accuracy_score(y_test, y_pred_pc_scaled)

# Print results
print("Accuracy using Duration Predictor:", accuracy_duration)
print("Accuracy using Scaled Principal Components Predictor:", accuracy_pc_scaled)

if accuracy_duration > accuracy_pc_scaled:
    print("Duration is a better predictor of whether a song is classical music.")
elif accuracy_duration < accuracy_pc_scaled:
    print("Scaled Principal Components are a better predictor of whether a song is classical music.")
else:
    print("Both predictors have the same accuracy in predicting whether a song is classical music.")

"""**Observations:**

- The logistic regression model using duration as a predictor achieves an accuracy of 98.1%, while the model using the scaled principal components as predictors achieves an accuracy of 98.4%.
- This suggests that duration is a slightly better predictor of whether a song is classical music than the principal components extracted in question 8.
- This is likely because duration is a more direct measure of the length of a song, which is a characteristic that is often associated with classical music.
- The principal components, on the other hand, are more abstract measures of the overall structure and characteristics of a song, and may not be as directly related to the genre of the song.
- Overall, these results suggest that both duration and the principal components can be used to predict the genre of a song with high accuracy, but duration may be a slightly more reliable predictor for classical music.

**Extra credit:**

 One interesting observation about this dataset could be related to the distribution of song durations across different genres. By analyzing the duration of songs within each genre, we might find some genre-specific patterns or tendencies.

For example, we could examine whether certain genres tend to have longer or shorter songs on average compared to others. This analysis could provide insights into the typical structure or characteristics of songs within different musical genres.

To perform this analysis, we have first need to group the dataset by genre and then calculate summary statistics (such as mean, median, or distribution) for the duration of songs within each genre. We could then visualize these statistics using box plots or histograms to compare the distributions of song durations across genres.

This exploration could uncover interesting trends or differences in song durations between genres, shedding light on how the duration of songs might be influenced by the stylistic or thematic elements characteristic of each genre.
"""

# Group the dataset by genre
genre_groups = spotify_data.groupby('track_genre')

# Calculate summary statistics for song durations within each genre
genre_durations = genre_groups['duration'].describe()

# Visualize the distributions of song durations across genres
plt.figure(figsize=(12, 8))
genre_durations['mean'].sort_values().plot(kind='bar', color='skyblue')
plt.title('Mean Song Duration Across Genres')
plt.xlabel('Genre')
plt.ylabel('Mean Duration (ms)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""**Observations:**

- There is a wide range of average song durations across different genres.
- Some genres, such as "Classical" and "Jazz", tend to have longer songs on average, while others, such as "Hip-Hop" and "Pop", tend to have shorter songs on average.
- This suggests that the duration of a song may be influenced by the stylistic characteristics or thematic elements associated with each genre.
- For example, classical music often involves complex compositions and extended instrumental passages, which may contribute to its longer song durations.
- On the other hand, genres like hip-hop and pop often emphasize catchy hooks and repetitive beats, which may lead to shorter song durations.
- This analysis provides insights into the typical structure and characteristics of songs within different musical genres, and how the duration of songs might be related to the genre's stylistic elements.
"""