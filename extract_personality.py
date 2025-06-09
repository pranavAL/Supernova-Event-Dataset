import os
import json
import pandas as pd

def extract_model_personalities(json_file_path):
    """
    Extract model names and personalities from a JSON file with events and models.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        pd.DataFrame: DataFrame containing Event, Model, and Model_Personality columns
    """
    try:
        # Load the JSON data
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Create a list to store the extracted information
        results = []
        
        # Iterate through each event in the data
        for event_data in data:
            event_name = event_data.get("Event", "Unknown Event")
            
            # Extract model information for this event
            for model_data in event_data.get("Models", []):
                model_name = model_data.get("Model", "Unknown Model")
                model_personality = model_data.get("Model_Personality", "Unknown Personality")
                
                # Add the extracted information to results
                results.append({
                    "Event": event_name,
                    "Model": model_name,
                    "Model_Personality": model_personality
                })
        
        # Convert the results to a DataFrame
        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        print(f"Error processing the JSON file: {e}")
        return None

def main():
    
    json_paths = {
        "movies":    "movies_results/movie_scripts_analysis.json",
        "biography": "biography_data_results/biography_analysis.json",
        "history":   "history_data_results/historical_events_analysis.json",
        "news":      "news_data_results/news_events_analysis.json",
    }

    csv_file_path = "model_personalities.csv"
    dfs = []

    # Pull results from each category
    for category, path in json_paths.items():
        df = extract_model_personalities(path)
        if df is not None and not df.empty:
            df["Category"] = category           # track the source category
            dfs.append(df)

    if not dfs:                      # nothing parsed
        print("No results to write.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # If a master CSV already exists, append new rows (deduplicating)
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        combined_df = (
            pd.concat([existing_df, combined_df], ignore_index=True)
              .drop_duplicates()
        )

    combined_df.to_csv(csv_file_path, index=False)
    print(f"Saved {len(combined_df)} rows to {csv_file_path}")

if __name__ == "__main__":
    main()