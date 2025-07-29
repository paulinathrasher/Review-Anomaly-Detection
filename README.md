# Anomaly Detection in Amazon E-commerce Reviews (Beauty Category)

This project applies two anomaly detection methods to Amazon product reviews using the **All_Beauty** category from the [McAuley et al. Amazon Reviews dataset (2023)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). It was developed as part of a Data Engineering course and includes data exploration, preprocessing, model implementation, and business insights.

## Dataset Overview

- **Category**: All_Beauty
- **Rows**: 701,528
- **Columns**: 12 total, including:
  - `review_id`, `rating`, `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, `verified_purchase`, `review_date`
- **Key Stats**:
  - Mean rating: 3.96 (75% = 5 stars)
  - Mean review length: ~33 words (std = 46)
  - Median helpful votes = 0 (highly skewed)

## Preprocessing

- Dropped rows with missing review text
- Filled missing `helpful_vote` with 0
- Cleaned text: lowercased, removed non-letter characters, tokenized, removed stopwords (NLTK)
- Feature Engineering:
  - `length_per_star`, `helpful_per_word`, `helpful_per_star`
- Normalized numeric features using `np.log1p`
- Vectorized text using **TF-IDF** with L2 normalization
- Generated sentence embeddings using **MiniLM** models

## Methods

### 1. Statistical Method – Z-Score Detection
- Computed multivariate Z-distance using Euclidean norm
- Flagged top 1% of reviews as anomalies
- Captured reviews with:
  - Very short yet highly voted content
  - Long, low-engagement reviews

### 2. Clustering-Based Method – K-Means on Embeddings
- Used **sentence-transformers** (`all-MiniLM-L6-v2` and `paraphrase-MiniLM-L12-v2`)
- Applied **PCA** (50 components) before clustering
- Selected `k=2` based on **silhouette score**
- Flagged top 1% distant reviews as anomalies
- Improved silhouette score with higher-quality embeddings:
  - From 0.0663 → 0.1297

## Evaluation & Comparison

| Criteria              | Statistical (Z-Score)            | Clustering (K-Means)               |
|-----------------------|----------------------------------|-------------------------------------|
| Type of anomalies     | Numeric outliers                 | Semantic outliers                   |
| Interpretability      | High (transparent rules)         | Medium (embedding & PCA dependent) |
| Flagged examples      | Short reviews with high votes    | Vague or linguistically odd reviews |
| Sensitivity           | ±3 z-score, ~2.74% flagged       | Top 1% furthest from centroid       |

- **Dimensionality Tuning**: PCA with 50 components outperformed 100
- **Sentence Embeddings**: Higher-quality models improve anomaly separation

## Business Insights

- **Short reviews with inflated helpful votes** may distort rankings and mislead users
- **Vague, semantically distinct reviews** reduce review quality and trust
- **Implications**:
  - Re-weight reviews in recommender systems
  - Flag outliers for human moderation
  - Reduce algorithmic bias in recommendation and fraud detection pipelines

## Example Anomalies

| Review Text          | Helpful Votes | Notes                              |
|----------------------|----------------|------------------------------------|
| "I like it"          | 172            | 3 words, high engagement           |
| "Smell"              | 1              | Too short / vague                  |
| "No sunscreen!!!!"   | 0              | Emotionally extreme, low context   |
| "Very cheap"         | 0              | Ambiguous meaning                  |

## Tech Stack

- **Python 3.10+**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `nltk`, `sentence-transformers`
- `PCA`, `KMeans`, `TF-IDF`, `Z-Score`, `Euclidean Distance`

## Citations

1. McAuley, J. et al. (2023). [Amazon Reviews Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
2. Reimers & Gurevych (2019). Sentence-BERT. [arXiv:1908.10084](http://arxiv.org/abs/1908.10084)
3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python
4. NLTK Project (2024). [Stopwords](https://www.nltk.org/search.html?q=stopwords)
5. Awan, A. (2024). [What is tokenization?](https://www.datacamp.com/blog/what-is-tokenization)
6. GeeksforGeeks (2025). [Understanding TF-IDF](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/)
7. Kaloyanova, E. (2024). [PCA & K-Means in Python](https://365datascience.com/tutorials/python-tutorials/pca-k-means/)
8. Murel & Kavlakoglu (2024). [Collaborative Filtering – IBM](https://www.ibm.com/think/topics/collaborative-filtering)

