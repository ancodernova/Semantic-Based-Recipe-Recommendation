
# Recipe Recommendation System

This project is a **Recipe Recommendation System** that leverages semantic search to recommend recipes based on user-provided ingredients. The model is built using **Sentence-BERT** embeddings and **FAISS** (Facebook AI Similarity Search) for fast and efficient nearest neighbor search. By encoding ingredient lists into embeddings, the system recommends recipes with similar ingredients, even when exact keyword matches don’t exist.

## Tech Stack

- **Python**: Core language for building the recommendation engine.
- **Sentence-BERT**: For embedding the ingredient lists into meaningful vectors.
- **FAISS**: Facebook AI’s library for efficient similarity search and clustering of dense vectors.
- **Scikit-Learn**: For metrics and train-test splitting.
- **Pandas**: For data manipulation and handling CSV files.
- **NumPy**: For efficient numerical computations, used extensively in FAISS and metric calculations.

## Features

- **Semantic Similarity**: Uses Sentence-BERT to embed ingredient lists into vectors, capturing the semantic meaning of the ingredients.
- **Fast Nearest Neighbor Search**: Utilizes FAISS for scalable and efficient vector similarity search.
- **Customizable Recommendations**: Allows users to input any set of ingredients and receive top-k recipe recommendations.
- **Model Evaluation**: Computes metrics like Precision, Recall, F1-score, Accuracy, RMSE, MSE, and MAE to measure model performance.

## Dataset

The model uses a recipe dataset that includes **Title** and **Ingredients** columns:
- `Ingredients`: A list of ingredients for each recipe.
- `Title`: The name of the recipe.

For the purposes of this README, assume the dataset file is named `recipes.csv`.

## Project Structure

```
recipe-recommendation/
├── data/
│   └── recipes.csv                   # Sample dataset with recipes and ingredients
├── models/
│   └── faiss_index.bin               # Precomputed FAISS index file for recipes
├── README.md                         # Project documentation
├── requirements.txt                  # List of dependencies
├── main.py                           # Main script to run the recommendation model
└── evaluate.py                       # Script for evaluating model performance
```

## Installation

### 1. Clone the Repository

To clone the repository, run:

```bash
git clone https://github.com/yourusername/recipe-recommendation.git
cd recipe-recommendation
```

### 2. Install Requirements

Make sure you have Python 3.7+ installed. Install the required packages using:

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Sentence-BERT Model

To use Sentence-BERT for encoding ingredients, download a pretrained model from [Hugging Face](https://huggingface.co/sentence-transformers) and adjust the model path in `main.py` if necessary.

## Usage

### 1. Prepare the Dataset

Place your dataset file (`recipes.csv`) in the `data/` directory. Ensure the dataset has at least two columns named `Ingredients` and `Title`.

### 2. Run the Model

You can start the recommendation system by running the `main.py` script:

```bash
python main.py
```

### 3. Example Usage

The model allows users to input a list of ingredients, which it then uses to recommend recipes. Here’s an example of how you might call the recommendation function in `main.py`:

```python
from main import recommend_recipes

# User's input ingredients
user_ingredients = "bread,jam"

# Get recipe recommendations
recommendations, distances = recommend_recipes(user_ingredients)

print("Recommended Recipes:", recommendations)
print("Distances:", distances)
```

### 4. Evaluation

The evaluation script (`evaluate.py`) assesses model performance by computing Precision, Recall, F1-score, Accuracy, RMSE, MSE, and MAE metrics. This script splits the dataset into training and testing sets and iteratively evaluates the recommendations.

To run the evaluation:

```bash
python evaluate.py
```

## Evaluation Metrics

The model’s performance is evaluated using the following metrics:

- **Precision**: Proportion of relevant recipes among those recommended.
- **Recall**: Proportion of relevant recipes retrieved out of all relevant recipes.
- **F1-score**: Harmonic mean of Precision and Recall.
- **Accuracy**: Proportion of times the true recipe is included in the recommendations.
- **RMSE (Root Mean Squared Error)**: Measures deviation in similarity scores for the closest recommendations.
- **MSE (Mean Squared Error)**: Similar to RMSE but without the square root, helpful for identifying large deviations.
- **MAE (Mean Absolute Error)**: Shows average distance, making it more interpretable than MSE.

## Troubleshooting

- **ValueError**: If you encounter a "ValueError: too many values to unpack" error, ensure the `recommend_recipes` function is correctly defined to return both `recommendations` and `distances`.
- **Embedding Model Error**: Make sure the Sentence-BERT model path is correctly specified if you're loading a custom model.

## Customization

You can customize various parameters within the `main.py` and `evaluate.py` scripts:
- **Top-K Recommendations**: Adjust the number of top recipes returned by changing the `top_k` parameter in `recommend_recipes`.
- **Evaluation Split**: Modify the test size ratio in `evaluate.py` if you want to use a different dataset split for evaluation.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments

- **Sentence-BERT** from [Hugging Face](https://huggingface.co/sentence-transformers) for the embeddings.
- **FAISS** by Facebook AI for efficient similarity search.


