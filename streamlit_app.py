import streamlit as st
import json
import requests
import pandas as pd
import time
import random


# Constants
MAX_RETRIES = 3  # Maximum number of retries for the API call
BACKOFF_FACTOR = 2  # Factor to determine the backoff period

# Set up OpenAI
openai_api_key = st.secrets["openai_api_key"]
pinecone_api_key = st.secrets["pinecone_api_key"]

def get_openai_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }
    data = {"input": text, "model": "text-embedding-ada-002"}

    for attempt in range(MAX_RETRIES):
        response = requests.post(
            "https://api.openai.com/v1/embeddings", headers=headers, json=data
        )

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        elif response.status_code == 429 or response.status_code >= 500:
            # If rate limited or server error, wait and retry
            time.sleep((BACKOFF_FACTOR**attempt) + random.uniform(0, 1))
        else:
            # For other errors, raise exception without retrying
            raise Exception(f"OpenAI API error: {response.text}")
    raise Exception(f"OpenAI API request failed after {MAX_RETRIES} attempts")


# Load treatments data
with open("treatments.json", "r") as file:
    treatments_data = json.load(file)


# Streamlit UI
st.set_page_config(page_title="Treatment Search Engine", page_icon=":mag:", layout="wide")

# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image(
        "https://usercontent.one/wp/www.shift-construction.com/wp-content/uploads/2023/10/shift-grey-logo-white-text_small.png",
        width=100,
    )
with col2:
    st.title("Treatment Search Engine")
with st.form(key="search_form"):
    search_query = st.text_input("Enter a search term", key="search_box")
    submit_button = st.form_submit_button(label="Search")

if submit_button and search_query:
    # Get embedding from OpenAI
    embedding = get_openai_embedding(search_query)


    # Pinecone query
    pinecone_response = requests.post(
        "https://uniclass-a569a40.svc.us-west4-gcp-free.pinecone.io/query",
        json={
            "vector": embedding,
            "topK": 10,
            "includeValues": False,
            "includeMetadata": True,
            "namespace": "hse",
        },
        headers={"Api-Key": pinecone_api_key},
    ).json()



    # Extract matching IDs
    matches = pinecone_response.get("matches", [])
    match_ids = [int(match["id"]) for match in matches]
    match_sentences = [match["metadata"]["sentence"] for match in matches]

    # Find matching treatments in the loaded data
    matching_treatments = [
        treatment
        for treatment in treatments_data
        if treatment["ScenarioID"] in match_ids
    ]

    # Display results
    if matching_treatments:
        for match_id, match_sentence in zip(match_ids, match_sentences):
            # Filter treatments for the current match ID
            filtered_treatments = [
                treatment
                for treatment in matching_treatments
                if treatment["ScenarioID"] == match_id
            ]

            if filtered_treatments:
                # Convert the filtered treatments to a DataFrame

                df = pd.DataFrame(filtered_treatments)
                df = df.drop(["Details", "ScenarioID", "Id"], axis=1)
                df = df.pivot_table(
                    index="Stage",
                    columns="Type",
                    values="Title",
                    aggfunc=lambda x: "<br>".join(x),
                    fill_value="",
                )

                # Create an expander for each matching ID using the match_sentence as the label
                with st.expander(f"{match_sentence}"):
                    # Display the DataFrame as a table within the expander
                    st.write(df.to_html(escape=False), unsafe_allow_html=True)

        st.markdown(
            """
            <style>
            /* Set the table width to 100% to fill the container */
            .stDataFrame {
                width: 100%;
            }
            /* Set the first column width and enable word-wrap */
            .stDataFrame tbody tr th:first-child,
            .stDataFrame tbody tr td:first-child {
                width: 100px !important;
                max-width: 100px !important;
                min-width: 100px !important;
                word-wrap: break-word !important;
            }
            /* Set the other columns to auto width for equal distribution and enable word-wrap */
            .stDataFrame tbody tr th:not(:first-child),
            .stDataFrame tbody tr td:not(:first-child) {
                width: auto !important;
                max-width: none !important;
                min-width: auto !important;
                word-wrap: break-word !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.write("No matches found.")
