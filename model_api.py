import pandas as pd
import numpy as np
import pickle
import re
import os
import warnings
import sqlite3

# Import Stemmer to match the CSV columns (e.g. 'adventure' -> 'adventur')
try:
    import nltk
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: 'nltk' not found. Install it with 'pip install nltk' for better predictions.")

class CollectiveIntelligenceSystem:
    def __init__(self, model_path='movie_tag_classifier.pkl', 
                 binarizer_path='tag_binarizer.pkl', 
                 vocab_path='feature_vocabulary.pkl',
                 db_path='movies.db'):
        
        # 1. Load the ML Model Components
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f: self.model = pickle.load(f)
            with open(binarizer_path, 'rb') as f: self.mlb = pickle.load(f)
            with open(vocab_path, 'rb') as f: self.vocabulary = pickle.load(f)
            print(" -> AI Model loaded successfully.")
        else:
            print(f" -> WARNING: Model file not found at {model_path}.")
            self.model = None

        # 2. Set Database Path
        self.db_path = db_path
        if os.path.exists(db_path):
            print(f" -> Database found: {db_path}.")
        else:
            print(f" -> WARNING: Database not found at {db_path}.")

        # Tag Mapping
        self.tag_mapping = {
            1: "CampyHumor", 2: "ClaustrophobicSetting", 3: "DarkThemes",
            4: "DialogueHeavy", 5: "ExperimentalStyle", 6: "FeelGood",
            7: "GraphicViolence", 8: "Injustice", 9: "LonelinessThemes",
            10: "MorallyAmbiguous", 11: "NostalgicAppeal", 12: "PsychologicalIntensity",
            13: "SatiricalTone", 14: "SlowBurn", 15: "SocialConflict",
            16: "StylizedVisuals", 17: "TragicRomance", 18: "TwistEnding",
            19: "UnflinchingRealism", 20: "VisuallyStunning"
        }

    def _preprocess_review(self, text):
        """
        Cleans text, applies stemming, and converts to DataFrame for the model.
        """
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = clean_text.split()
        
        if NLTK_AVAILABLE:
            words = [stemmer.stem(w) for w in words]

        vector = [0] * len(self.vocabulary)
        for i, vocab_word in enumerate(self.vocabulary):
            if vocab_word in words:
                vector[i] = 1
        
        return pd.DataFrame([vector], columns=self.vocabulary)

    def predict_tags_from_review(self, review_text):
        """
        Takes a review string -> Returns list of predicted Tag Names
        """
        if not self.model: return ["Error: Model not loaded"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            input_df = self._preprocess_review(review_text)
            
            # --- USE PROBABILITIES TO FORCE A PREDICTION ---
            try:
                # Get probabilities for all 20 tags
                probs = self.model.predict_proba(input_df)[0]
                
                # Pair scores with Tag IDs
                scored_tags = []
                for idx, score in enumerate(probs):
                    tag_id = self.mlb.classes_[idx]
                    scored_tags.append((score, tag_id))
                
                # Sort: Highest confidence first
                scored_tags.sort(key=lambda x: x[0], reverse=True)
                
                result_ids = []
                
                # Rule 1: Always take the #1 highest scoring tag (even if low confidence)
                if len(scored_tags) > 0:
                    result_ids.append(scored_tags[0][1])
                
                # Rule 2: Also include #2 and #3 if they are decent (score > 0.15)
                for score, tag_id in scored_tags[1:3]:
                    if score > 0.15:
                        result_ids.append(tag_id)
                        
            except AttributeError:
                # Fallback if model doesn't support probabilities
                pred = self.model.predict(input_df)
                result_ids = self.mlb.inverse_transform(pred)[0]

        # Convert IDs to Names
        result_tags = []
        for tag_id in result_ids:
            name = self.tag_mapping.get(tag_id, str(tag_id))
            if name not in result_tags:
                result_tags.append(name)
        
        if not result_tags:
            return ["No specific tag detected"]
            
        return result_tags

    def recommend_movies_by_tags(self, selected_tags):
        """
        Takes list of Tag Names -> Returns list of matching Movie Reviews
        """
        if not os.path.exists(self.db_path): return ["Error: Database not found"]
        if not selected_tags: return ["No tags selected."]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create placeholders for IN clause
            placeholders = ','.join(['?'] * len(selected_tags))
            
            # Query to find movies that match ANY of the selected tags
            query = f"""
                SELECT title, year, description 
                FROM movies 
                WHERE tag1 IN ({placeholders}) 
                   OR tag2 IN ({placeholders}) 
                   OR tag3 IN ({placeholders})
                LIMIT 5
            """
            
            # Parameters need to be repeated for each IN clause
            params = selected_tags * 3
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return ["No movies found for these tags."]
            
            # Format results
            results = []
            for r in rows:
                title, year, desc = r
                results.append(f"**{title} ({year})**\n{desc}")
                
            return results

        except Exception as e:
            return [f"Database Error: {e}"]

# ==========================================
#  TESTING AREA
# ==========================================
if __name__ == "__main__":
    print("\n--- FINAL TEST RUN ---")
    
    # 1. Initialize System
    system = CollectiveIntelligenceSystem()
    
    # TEST 1: Prediction
    print("\n[TEST 1] Predicting Tags for a Review...")
    test_review = "This movie was incredibly dark, scary, and full of violence."
    predicted = system.predict_tags_from_review(test_review)
    print(f"Review: '{test_review}'")
    print(f"Predicted: {predicted}")

    # TEST 2: Recommendation
    print("\n[TEST 2] Finding Movies for a Tag...")
    target_tag = "CampyHumor"
    recommendations = system.recommend_movies_by_tags([target_tag])
    print(f"Searching for: {target_tag}")
    print(f"Found {len(recommendations)} movies.")
    
    if len(recommendations) > 0:
        print(f"First Recommendation Snippet: {recommendations[0][:100]}...")
    
    print("\n--- TEST RUN COMPLETE ---")