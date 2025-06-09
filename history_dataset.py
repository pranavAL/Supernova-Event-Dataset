import os
import re
import logging
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
import ollama
import pandas as pd
import time
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)

# Argument parsing
parser = argparse.ArgumentParser(description="Run LLM analysis pipeline")
parser.add_argument(
    "--model",
    type=str,
    choices=["phi4", "orca2:13b", "qwen2.5:14b"],
    default="qwen2.5:14b",
    help="Name of the model to use"
)
args = parser.parse_args()

# Constants
MODELS = args.model
EMBEDDING_MODEL = "nomic-embed-text"
ANALYSIS_MODEL = "qwen2.5:14b"  # Model used for personality analysis
VECTOR_STORE_DIR = "vector_stores"
RESULTS_DIR = "history_data_results"
OUTPUT_FILE = "historical_events_analysis.json"
HISTORICAL_EVENTS_DIR = "historical-events"

def ensure_directories():
    """Ensure necessary directories exist."""
    for directory in [VECTOR_STORE_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    logging.info("Directories created.")

def process_text_file(filepath):
    """Process a text file and return document."""
    try:
        loader = TextLoader(filepath)
        documents = loader.load()
        logging.info(f"Successfully loaded file: {filepath}")
        return documents
    except Exception as e:
        logging.error(f"Error processing {filepath}: {str(e)}")
        return None

def load_historical_events(directory=HISTORICAL_EVENTS_DIR, event_name=None):
    """Load historical event text files."""
    if not os.path.exists(directory):
        logging.error(f"Directory not found: {directory}")
        return [], []
    
    documents = []
    event_titles = []
    
    if event_name:
        # Process a single specific event
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                event_title_from_file = os.path.splitext(filename)[0].replace('-', ' ').title()
                if event_title_from_file.lower() == event_name.lower():
                    file_path = os.path.join(directory, filename)
                    docs = process_text_file(file_path)
                    if docs:
                        documents.extend(docs)
                        event_titles.append(event_name)
                    break
    else:
        # Process all events in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                docs = process_text_file(file_path)
                if docs:
                    documents.extend(docs)
                    # Store event title formatted from filename
                    event_title = os.path.splitext(filename)[0].replace('-', ' ').title()
                    event_titles.append(event_title)
    
    if not documents:
        logging.error("No text files found or loaded")
        return [], []
    
    logging.info(f"Successfully loaded {len(documents)} historical event files")
    return documents, event_titles

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

def create_vector_db(chunks, event_title, model_name):
    """Create a vector database from document chunks."""
    ollama.pull(EMBEDDING_MODEL)
    # Sanitize collection name to comply with Chroma's requirements
    sanitized_model_name = model_name.replace(":", "_").replace(".", "_")
    sanitized_event_title = event_title.replace(" ", "_").lower()
    collection_name = f"{sanitized_event_title}_{sanitized_model_name}"
    vector_db_path = os.path.join(VECTOR_STORE_DIR, collection_name)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=collection_name,
        persist_directory=vector_db_path
    )
    logging.info(f"Vector database created for {event_title} with {model_name}")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever focused on identifying and ranking exactly 5 critical events."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are analyzing the following question: {question}

    Your task is to extract exactly **5 critical events** from the provided historical account. Identify key events or milestones that significantly shaped the course of history or had major impacts.

    Instructions:
    - Identify exactly 5 key events or milestones.
    - For each event, consider factors such as:
        - Did the event significantly alter the course of history?
        - Was it a turning point that led to a different trajectory?
        - If this event had not occurred, would the overall outcome be drastically different?
        - Did the event have long-term consequences that shaped future events?
    - Rank the 5 events from **most critical to least critical** based on their cascading effects.
    - Provide a concise summary of each event in one or at most two sentences.

    Return a ranked list of the 5 critical events with their summaries.
    """,
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain for processing queries."""
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}

    Please provide your reasoning step-by-step before giving the final answer. In your reasoning:
    1. Identify exactly 5 critical events or milestones from the historical account.
    2. Rank these events from **most critical to least critical** based on their impact or cascading effects on history.
    3. For each event, provide a concise summary in one or two sentences.
    4. Explain why the top-ranked event is the most critical, including:
    - How it had a cascading effect on subsequent historical developments.
    - How history would have been different if this event had not occurred.
    - Why this event stands out as the most critical.

    Your final answer should include:
    - A ranked list of exactly 5 critical events with their summaries.
    - A clear statement of the most critical event and its impact.
    """
    prompt = PromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created successfully.")
    return chain

def extract_critical_events(response, model_name):
    """Extract critical events from the response based on the model."""
    # Initialize variables
    critical_events = []
    top_event_explanation = ""
    
    # Different extraction logic based on model
    if model_name == "phi4":
        # Extract numbered events with asterisks (phi4 style)
        event_pattern = r"\d+\.\s+\*\*.*?\*\*.*?(?=\n\d+\.|\n\n|$)"
        critical_events = re.findall(event_pattern, response, re.DOTALL)
        
        # Look for Why Critical or Cascading Effect section
        explanation_pattern = r"(?:why this event stands out|why this event is|cascading effect|most critical).*?(?=\n\n|$)"
        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            top_event_explanation = explanation_match.group(0).strip()
    
    elif model_name == "orca2:13b":
        # First, find the actual ranked list section
        ranked_list_section = ""
        ranked_pattern = r"(?:ranked list|the five critical|events is).*?(?:\d+\.\s+.*(?:\n|$))+"
        ranked_match = re.search(ranked_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if ranked_match:
            ranked_list_section = ranked_match.group(0)
            # Extract the numbered events
            event_pattern = r"\d+\.\s+.*?(?=\n\d+\.|\n\n|$)"
            critical_events = re.findall(event_pattern, ranked_list_section, re.DOTALL)
        
        # Extract explanation for top event
        explanation_pattern = r"most critical event is.*?because:.*?(?=\n\n|$)"
        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            top_event_explanation = explanation_match.group(0).strip()
    
    elif model_name == "qwen2.5:14b":
        # First find the "Ranking of Events" or "Summaries and Analysis" section
        ranked_section = ""
        summary_pattern = r"(?:ranking of|list of|the critical).*?(?:\d+\.\s+.*(?:\n|$))+"
        summary_match = re.search(summary_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if summary_match:
            ranked_section = summary_match.group(0)
            # Extract each numbered event
            event_pattern = r"\d+\.\s+.*?(?=\n\d+\.|\n\n|$)"
            critical_events = re.findall(event_pattern, response, re.DOTALL)
        
        # Extract explanation for why the top event is critical
        explanation_pattern = r"(?:why it\'s most critical|most critical event|cascading effect).*?(?=\n\n|$)"
        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            top_event_explanation = explanation_match.group(0).strip()
    
    # Default extraction if the model-specific extraction fails
    if not critical_events:
        # Try a more general pattern
        event_pattern = r"\d+\.[\s\*]*[^0-9\n]+\n"
        events = re.findall(event_pattern, response)
        critical_events = events[:5]  # Take just the first 5 events
    
    # Clean up the extracted events
    critical_events = [event.strip() for event in critical_events if event.strip()]
    
    # Ensure we have only 5 events (or fewer if not enough were found)
    critical_events = critical_events[:5]
    
    # If we didn't find a good explanation, try a more general search
    if not top_event_explanation:
        general_patterns = [
            r"most critical.*?because.*?(?=\n\n|$)",
            r"critical as it.*?(?=\n\n|$)",
            r"impact.*?historical.*?(?=\n\n|$)",
            r"significance.*?(?=\n\n|$)"
        ]
        
        for pattern in general_patterns:
            explanation_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if explanation_match:
                top_event_explanation = explanation_match.group(0).strip()
                break
    
    # Return the top event and its explanation
    if critical_events:
        return critical_events, top_event_explanation
    else:
        # If no events were extracted, return a placeholder
        return ["No critical events identified"], ""

def llm_determined_personality(full_response, critical_events, model_name, event_name):
    """Use an LLM to determine model personality based on its output."""
    # Create the prompt for personality analysis
    prompt = f"""You are evaluating the personality of an AI model based purely on the **themes and focus** of the critical events it selects from a historical event. **Do not** classify based on reasoning style (e.g., "Logical", "Analytical", "Methodical"). Instead, focus only on the nature of the events themselves.
    
    Below is a response from model "{model_name}" when asked to identify and rank 5 critical developments in the news event "{event_name}". **Analyze only the content and themes of the selected events** and classify the personality accordingly.
    
    ---BEGIN MODEL RESPONSE---
    {full_response}  
    ---END MODEL RESPONSE---
    
    Your response should be exactly ONE LINE with just the personality classification. **Do not** classify based on reasoning style (e.g., "Logical", "Analytical", "Methodical", "Critical")
    
    """
    
    # Call a separate LLM to analyze the model personality
    
    # Initialize a separate LLM for analysis
    logging.info(f"Using {ANALYSIS_MODEL} to analyze model personality")
    analysis_llm = ChatOllama(model=ANALYSIS_MODEL)
    
    # Get personality analysis (with retry logic)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            personality = analysis_llm.invoke(prompt)
            personality = personality.content
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Retry attempt {attempt+1} for personality analysis: {str(e)}")
                time.sleep(2)  # Short delay before retry
            else:
                raise e
    
    return personality        

def analyze_historical_event(event_docs, event_title):
    """Analyze a historical event."""
    logging.info(f"Analyzing historical event: {event_title}")
    
    # Create a sanitized filename version of the event title
    event_filename = event_title.replace(" ", "_").lower()
    
    # Create a result object for this event
    event_results = {"Event": event_title, "Models": []}
    
    # Process with each model
    for model_name in MODELS:
        # Check if we already have results for this event+model combination
        result_file = os.path.join(RESULTS_DIR, f"{event_filename}_{model_name.replace(':', '_')}.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    model_result = json.load(f)
                    logging.info(f"Loaded existing results for {event_title} with {model_name}")
                    event_results["Models"].append(model_result)
                    continue
            except Exception as e:
                logging.warning(f"Could not load existing result, will reprocess: {e}")
        
        logging.info(f"Using model: {model_name}")
        
        try:
            # Initialize the language model
            llm = ChatOllama(model=model_name)
            
            # Split the documents into chunks
            chunks = split_documents(event_docs)

            # Create the vector database
            vector_db = create_vector_db(chunks, event_title, model_name)

            # Create the retriever
            retriever = create_retriever(vector_db, llm)

            # Create the chain
            chain = create_chain(retriever, llm)

            # Define the question
            question = f"Identify exactly 5 critical events or milestones from the historical account of {event_title}?"

            # Get the response
            response = chain.invoke(input=question)
            logging.info(f"Response for {event_title}: {response}")

            # Extract critical events and explanations
            critical_events, top_event_explanation = extract_critical_events(response, model_name)
            
            # Determine model personality using LLM analysis
            model_personality = llm_determined_personality(response, critical_events, model_name, event_title)
            logging.info(f"Determined model personality: {model_personality}")
            
            # Make sure critical_events has at least one item
            if not critical_events:
                critical_events = ["No critical events identified"]
            
            # Prepare model result
            model_result = {
                "Model": model_name,
                "Model_Personality": model_personality,
                "Top_Critical_Event": critical_events[0],
                "Why_Critical": top_event_explanation,
                "Other_Key_Events": critical_events[1:5] if len(critical_events) > 1 else [],
                "Full_Response": response
            }
            
            event_results["Models"].append(model_result)
            
            # Save intermediate results to avoid losing progress
            sanitized_model_name = model_name.replace(':', '_')
            with open(os.path.join(RESULTS_DIR, f"{event_filename}_{sanitized_model_name}.json"), "w") as f:
                json.dump(model_result, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error processing {event_title} with {model_name}: {str(e)}")
            error_result = {
                "Model": model_name,
                "Model_Personality": "Error",
                "Top_Critical_Event": "Processing failed",
                "Why_Critical": f"Error: {str(e)}",
                "Other_Key_Events": [],
                "Full_Response": f"Error during processing: {str(e)}"
            }
            event_results["Models"].append(error_result)
    
    # Save the full event results
    with open(os.path.join(RESULTS_DIR, f"{event_filename}_analysis.json"), "w") as f:
        json.dump(event_results, f, indent=4)
    
    # Also update the full output file
    combined_dataset_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    if os.path.exists(combined_dataset_path):
        try:
            with open(combined_dataset_path, 'r') as f:
                combined_dataset = json.load(f)
                
            # Check if this event is already in the dataset
            event_exists = False
            for i, event in enumerate(combined_dataset):
                if event["Event"] == event_title:
                    combined_dataset[i] = event_results
                    event_exists = True
                    break
            
            if not event_exists:
                combined_dataset.append(event_results)
                
            with open(combined_dataset_path, 'w') as f:
                json.dump(combined_dataset, f, indent=4)
        except:
            # If there's an error reading or updating, just overwrite with a new dataset
            with open(combined_dataset_path, 'w') as f:
                json.dump([event_results], f, indent=4)
    else:
        # If the file doesn't exist, create it
        with open(combined_dataset_path, 'w') as f:
            json.dump([event_results], f, indent=4)
    
    return event_results

def analyze_all_historical_events():
    """Analyze all historical events in the directory."""
    ensure_directories()
    
    # Load all historical events
    documents, event_titles = load_historical_events()
    
    if not documents:
        logging.error("No historical event documents to analyze")
        return None
    
    # Analyze each event
    all_results = []
    for i, (doc, event_title) in enumerate(zip(documents, event_titles)):
        logging.info(f"Processing event {i+1}/{len(documents)}: {event_title}")
        event_result = analyze_historical_event([doc], event_title)
        all_results.append(event_result)
    
    # Generate reports
    generate_reports()
    
    return all_results

def analyze_specific_historical_event(event_name):
    """Analyze a specific historical event by name."""
    ensure_directories()
    
    # Load the specific historical event
    documents, event_titles = load_historical_events(event_name=event_name)
    
    if not documents:
        logging.error(f"Historical event '{event_name}' not found or failed to load")
        return None
    
    # Analyze the event
    event_result = analyze_historical_event(documents, event_titles[0])
    
    # Generate reports
    generate_reports()
    
    return event_result

def generate_reports():
    """Generate comparative reports for the results."""
    # Read the combined dataset
    try:
        with open(os.path.join(RESULTS_DIR, OUTPUT_FILE), 'r') as f:
            all_results = json.load(f)
    except:
        logging.error("Could not load combined dataset for reports")
        return
    
    # Create a DataFrame for easier analysis
    rows = []
    for event in all_results:
        event_title = event["Event"]
        for model_result in event["Models"]:
            row = {
                "Event": event_title,
                "Model": model_result["Model"],
                "Model_Personality": model_result["Model_Personality"],
                "Top_Critical_Event": model_result["Top_Critical_Event"],
                "Why_Critical": model_result["Why_Critical"]
            }
            
            # Add other events
            for i, other_event in enumerate(model_result["Other_Key_Events"], 2):
                row[f"Event_{i}"] = other_event
                
            rows.append(row)
    
    if not rows:
        logging.error("No data available for reports")
        return
        
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(os.path.join(RESULTS_DIR, "historical_events_analysis_summary.csv"), index=False)
    
    # Generate model personality statistics
    personality_stats = df.groupby(['Model', 'Model_Personality']).size().reset_index(name='Count')
    personality_stats.to_csv(os.path.join(RESULTS_DIR, "historical_model_personality_stats.csv"), index=False)
    
    # Generate HTML report
    generate_html_report(df)
    
    # Generate model personality comparison table
    create_model_personality_table()
    
    logging.info("Reports generated successfully")

def generate_html_report(df):
    """Generate an HTML report with visualizations."""
    # Create a basic HTML report with tables and descriptions
    event_count = df['Event'].nunique()
    model_count = df['Model'].nunique()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Historical Events Critical Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .model-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .model-header {{ background-color: #f2f2f2; padding: 10px; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Historical Events Critical Analysis</h1>
        <p>Analyzed {event_count} historical events using {model_count} different models with LLM-based personality detection.</p>
        
        <h2>Model Personality Distribution</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Personality</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
    """
    
    # Add model personality statistics
    personality_stats = df.groupby(['Model', 'Model_Personality']).size().reset_index(name='Count')
    for model in df['Model'].unique():
        model_rows = personality_stats[personality_stats['Model'] == model]
        total = model_rows['Count'].sum()
        for _, row in model_rows.iterrows():
            percentage = (row['Count'] / total) * 100
            html_content += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['Model_Personality']}</td>
                <td>{row['Count']}</td>
                <td>{percentage:.2f}%</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h2>Sample Historical Event Analysis</h2>
    """
    
    # Add sample event analyses
    for event in df['Event'].unique()[:5]:  # Show first 5 events
        event_data = df[df['Event'] == event]
        html_content += f"""
        <h3>{event}</h3>
        """
        
        for _, row in event_data.iterrows():
            html_content += f"""
            <div class="model-card">
                <div class="model-header">
                    <strong>Model:</strong> {row['Model']} ({row['Model_Personality']})
                </div>
                <p><strong>Top Critical Event:</strong> {row['Top_Critical_Event']}</p>
                <p><strong>Why Critical:</strong> {row['Why_Critical']}</p>
                <p><strong>Other Key Events:</strong></p>
                <ol start="2">
            """
            
            for i in range(2, 6):
                event_key = f"Event_{i}"
                if event_key in row and not pd.isna(row[event_key]):
                    html_content += f"<li>{row[event_key]}</li>"
            
            html_content += """
                </ol>
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(os.path.join(RESULTS_DIR, "historical_events_report.html"), "w") as f:
        f.write(html_content)

def create_model_personality_table():
    """Create a comparison table of model personalities."""
    try:
        with open(os.path.join(RESULTS_DIR, OUTPUT_FILE), 'r') as f:
            all_results = json.load(f)
            
        # Create a DataFrame to store the comparison
        comparison_data = []
        
        for event in all_results:
            event_title = event["Event"]
            
            # Extract model personalities for each model
            model_personalities = {}
            model_top_events = {}
            
            for model_result in event["Models"]:
                model_name = model_result["Model"]
                model_personalities[model_name] = model_result["Model_Personality"]
                model_top_events[model_name] = model_result["Top_Critical_Event"]
            
            # Add to comparison data
            comparison_data.append({
                "Event": event_title,
                **{f"{model}_Personality": model_personalities.get(model, "N/A") for model in MODELS},
                **{f"{model}_TopEvent": model_top_events.get(model, "N/A") for model in MODELS}
            })
        
        # Create the DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_df.to_csv(os.path.join(RESULTS_DIR, "historical_model_personality_comparison.csv"), index=False)
        
        # Create an HTML version of the table
        html_table = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Personality Comparison for Historical Events</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Model Personality Comparison Across Historical Events</h1>
            <p>Using LLM-based personality detection</p>
            <table>
                <tr>
                    <th>Event</th>
        """
        
        for model in MODELS:
            html_table += f"""
                    <th>{model} Personality</th>
                    <th>{model} Top Event</th>
            """
        
        html_table += """
                </tr>
        """
        
        for _, row in comparison_df.iterrows():
            html_table += f"""
                <tr>
                    <td>{row['Event']}</td>
            """
            
            for model in MODELS:
                html_table += f"""
                    <td>{row[f'{model}_Personality']}</td>
                    <td>{row[f'{model}_TopEvent']}</td>
                """
            
            html_table += """
                </tr>
            """
        
        html_table += """
            </table>
        </body>
        </html>
        """
        
        # Save the HTML table
        with open(os.path.join(RESULTS_DIR, "historical_model_personality_comparison.html"), "w") as f:
            f.write(html_table)
        
        return comparison_df
    except Exception as e:
        logging.error(f"Error creating model personality table: {str(e)}")
        return None

def main():
    """Main function to run the analysis."""
    # Example usage - analyze all historical events
    all_results = analyze_all_historical_events()

if __name__ == "__main__":
    main()