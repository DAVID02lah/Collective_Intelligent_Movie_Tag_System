import streamlit as st
import pandas as pd
from model_api import CollectiveIntelligenceSystem

# Initialize and cache the system
@st.cache_resource
def load_system():
    # This will load the model, binarizer, vocabulary, and database
    # It prints to console, but in Streamlit we just want the object
    return CollectiveIntelligenceSystem()

def main():
    st.set_page_config(page_title="Movie Tag System", layout="wide")
    
    st.title("🎬 Collective Intelligence: Movie Tag System")
    
    # Load the system
    with st.spinner("Loading AI Models..."):
        system = load_system()
    
    # Check if loaded correctly
    if not system.model:
        st.error("Error: Could not load model files. Please check 'model_api.py' and ensure .pkl/.csv files exist.")
        return

    # Sidebar Navigation
    page = st.sidebar.radio("Navigation", ["Recommend Movies", "Predict Tags from Review"])

    # -------------------------------------------------------------------------
    # PAGE 1: Recommend Movies
    # -------------------------------------------------------------------------
    if page == "Recommend Movies":
        st.header("🔍 Find Movies by Tags")
        st.write("Select up to 3 tags to find movie recommendations.")

        # Get all available tag names from the mapping
        all_tags = sorted(list(system.tag_mapping.values()))
        
        # User Selection
        selected_tags = st.multiselect("Select Tags:", all_tags, max_selections=3)
        
        if st.button("Recommend"):
            if not selected_tags:
                st.warning("Please select at least one tag.")
            else:
                with st.spinner("Searching database..."):
                    recommendations = system.recommend_movies_by_tags(selected_tags)
                
                st.subheader(f"Results ({len(recommendations)} found)")
                
                if recommendations and recommendations[0].startswith("No movies"):
                    st.info(recommendations[0])
                else:
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}. Recommendation:**")
                        st.write(f"> {rec}")
                        st.markdown("---")

    # -------------------------------------------------------------------------
    # PAGE 2: Predict Tags
    # -------------------------------------------------------------------------
    elif page == "Predict Tags from Review":
        st.header("🧠 AI Tag Predictor")
        st.write("Enter a movie review below, and the AI will detect its themes/tags.")

        user_review = st.text_area("Your Review:", height=150, placeholder="e.g., This movie was incredibly dark, scary, and full of violence...")

        if st.button("Predict Tags"):
            if not user_review.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing text..."):
                    predicted_tags = system.predict_tags_from_review(user_review)
                
                st.subheader("Predicted Tags:")
                
                if predicted_tags and predicted_tags[0].startswith("Error"):
                    st.error(predicted_tags[0])
                elif predicted_tags == ["No specific tag detected"]:
                    st.info("No specific tags detected with high confidence.")
                else:
                    # Display tags as chips/badges
                    cols = st.columns(len(predicted_tags))
                    for i, tag in enumerate(predicted_tags):
                        st.success(f"🏷️ {tag}")

if __name__ == "__main__":
    main()
