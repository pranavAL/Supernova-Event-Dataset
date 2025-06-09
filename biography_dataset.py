import os
import re
import logging
import json
import argparse
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
MODEL_NAME = args.model
EMBEDDING_MODEL = "nomic-embed-text"
ANALYSIS_MODEL = MODEL_NAME  # Use the model from the argument
VECTOR_STORE_DIR = "vector_stores"
RESULTS_DIR = "biography_data_results"
OUTPUT_FILE = "biography_analysis.json"
DISCOVERIES_DIR = "biographies"

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

def load_biography_datasets(directory=DISCOVERIES_DIR, biography_name=None):
    """Load biographies text files."""
    if not os.path.exists(directory):
        logging.error(f"Directory not found: {directory}")
        return [], []
    
    documents = []
    biography_titles = []
    
    if biography_name:
        # Process a single specific biographyt
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                biography_title_from_file = os.path.splitext(filename)[0].replace('-', ' ').title()
                if biography_title_from_file.lower() == biography_name.lower():
                    file_path = os.path.join(directory, filename)
                    docs = process_text_file(file_path)
                    if docs:
                        documents.extend(docs)
                        biography_titles.append(biography_name)
                    break
    else:
        # Process all discoveries in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                docs = process_text_file(file_path)
                if docs:
                    documents.extend(docs)
                    # Store biography title formatted from filename
                    biography_title = os.path.splitext(filename)[0].replace('-', ' ').title()
                    biography_titles.append(biography_title)
    
    if not documents:
        logging.error("No text files found or loaded")
        return [], []
    
    logging.info(f"Successfully loaded {len(documents)} biography documents.")
    return documents, biography_titles

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

def create_vector_db(chunks, biography_title, model_name):
    """Create a vector database from document chunks."""
    ollama.pull(EMBEDDING_MODEL)
    # Sanitize collection name to comply with Chroma's requirements
    sanitized_model_name = model_name.replace(":", "_").replace(".", "_")
    sanitized_biography_title = biography_title.replace(" ", "_").lower()
    collection_name = f"{sanitized_biography_title}_{sanitized_model_name}"
    vector_db_path = os.path.join(VECTOR_STORE_DIR, collection_name)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=collection_name,
        persist_directory=vector_db_path
    )
    logging.info(f"Vector database created for {biography_title} with {model_name}")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever focused on identifying and ranking exactly 5 critical events in a person's biography."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are analyzing the following question as an emotional person: {question}

    Your task is to extract exactly **5 critical moments or key events** from the provided biographical account. Identify pivotal life events, important decisions, major achievements, significant relationships, or transformative experiences that fundamentally shaped this person's life and legacy.

    Instructions:
    - Identify exactly 5 key milestones or pivotal moments in this person's life.
    - For each milestone, consider factors such as:
        - Did it fundamentally alter their life trajectory or career path?
        - Was it a turning point that led to new opportunities or directions?
        - If this event had not occurred, would their life story have been significantly different?
        - Did it reflect or demonstrate their core values, character, or unique contributions?
        - Did it have long-term impacts on their legacy, influence, or how they're remembered?
    - Rank the 5 milestones from **most impactful to least impactful** based on their biographical significance.
    - Provide a concise summary of each milestone in one or at most two sentences.

    Return a ranked list of the 5 critical milestones with their summaries.
    """,
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain for processing queries about biographical information."""
    template = """Answer the question based ONLY on the following context  as an emotional person:
    {context}

    Question: {question}

    Please provide your reasoning step-by-step before giving the final answer. In your reasoning:
    1. Identify exactly 5 critical milestones or key events from the person's biography.
    2. Rank these milestones from **most impactful to least impactful** based on their biographical significance.
    3. For each milestone, provide a concise summary in one or two sentences.
    4. Explain why the top-ranked milestone is the most impactful, including:
    - How it fundamentally shaped their life trajectory or legacy.
    - How their story would have unfolded differently if this event had not occurred.
    - Why this milestone stands out as the most biographically significant.

    Your final answer should include:
    - A ranked list of exactly 5 critical milestones with their summaries.
    - A clear statement of the most impactful milestone and its significance to the person's life and legacy.
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

def extract_critical_biographical_events(response, model_name):
    """Extract critical biographical events from the response based on the model."""
    # Initialize variables
    critical_events = []
    top_event_explanation = ""
    
    # Different extraction logic based on model
    if model_name == "phi4":
        # Extract numbered events with asterisks (phi4 style)
        event_pattern = r"\d+\.\s+\*\*.*?\*\*.*?(?=\n\d+\.|\n\n|$)"
        critical_events = re.findall(event_pattern, response, re.DOTALL)
        
        # Look for Why Impactful or Life-Changing section
        explanation_pattern = r"(?:why this milestone|why this event|fundamental impact|most impactful).*?(?=\n\n|$)"
        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            top_event_explanation = explanation_match.group(0).strip()
    
    elif model_name == "orca2:13b":
        # First, find the actual ranked list section
        ranked_list_section = ""
        ranked_pattern = r"(?:ranked list|the five critical|milestones is|events is|moments is).*?(?:\d+\.\s+.*(?:\n|$))+"
        ranked_match = re.search(ranked_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if ranked_match:
            ranked_list_section = ranked_match.group(0)
            # Extract the numbered events
            event_pattern = r"\d+\.\s+.*?(?=\n\d+\.|\n\n|$)"
            critical_events = re.findall(event_pattern, ranked_list_section, re.DOTALL)
        
        # Extract explanation for top event
        explanation_pattern = r"most (?:impactful|significant) milestone is.*?because:.*?(?=\n\n|$)"
        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            top_event_explanation = explanation_match.group(0).strip()
    
    elif model_name == "qwen2.5:14b":
        # First find the "Ranking of Events" or "Summaries and Analysis" section
        ranked_section = ""
        summary_pattern = r"(?:ranking of|list of|the critical|the impactful).*?(?:\d+\.\s+.*(?:\n|$))+"
        summary_match = re.search(summary_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if summary_match:
            ranked_section = summary_match.group(0)
            # Extract each numbered event
            event_pattern = r"\d+\.\s+.*?(?=\n\d+\.|\n\n|$)"
            critical_events = re.findall(event_pattern, response, re.DOTALL)
        
        # Extract explanation for why the top event is impactful
        explanation_pattern = r"(?:why it\'s most impactful|most impactful milestone|biographical significance|life trajectory|personal impact).*?(?=\n\n|$)"
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
            r"most impactful.*?because.*?(?=\n\n|$)",
            r"significant as it.*?(?=\n\n|$)",
            r"impact.*?life.*?(?=\n\n|$)",
            r"significance.*?legacy.*?(?=\n\n|$)",
            r"shaped.*?trajectory.*?(?=\n\n|$)"
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
        return ["No critical life events identified"], ""

def llm_determined_personality(full_response, critical_events, model_name, biography_name):
    """Use an LLM to determine model personality based on its output."""
    # Create the prompt for personality analysis
    prompt = f"""You are evaluating the personality of an AI model based purely on the **themes and focus** of the critical events it selects from a biography. **Do not** classify based on reasoning style (e.g., "Logical", "Analytical", "Methodical"). Instead, focus only on the nature of the events themselves.
    
    Below is a response from model "{model_name}" when asked to identify 5 critical milestones in the biography of {biography_name}. **Analyze only the content and themes of the selected events** and classify the personality accordingly.
    
    ---BEGIN MODEL RESPONSE---
    {full_response} 
    ---END MODEL RESPONSE---
        
    Your response should be exactly ONE LINE with just the personality classification into one of the following categories: "Idealogical", "Emotional", "Strategic", "Creative", "Observational", "Public Influence", "Community Support". **Do not** classify based on reasoning style (e.g., "Logical", "Analytical", "Methodical", "Critical")
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
            print(personality)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Retry attempt {attempt+1} for personality analysis: {str(e)}")
                time.sleep(2)  # Short delay before retry
            else:
                raise e
    
    return personality 
        
        

def analyze_biography(biography_docs, biography_title):
    """Analyze a biography."""
    logging.info(f"Analyzing biography of: {biography_title}")
    
    # Create a sanitized filename version of the biography
    biography_filename = biography_title.replace(" ", "_").lower()
    
    # Create a result object for this biography
    biography_results = {"Biography": biography_title, "Models": []}
    
    # Process with each model
    for model_name in MODELS:
        # Check if we already have results for this biography+model combination
        result_file = os.path.join(RESULTS_DIR, f"{biography_filename}_{model_name.replace(':', '_')}.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    model_result = json.load(f)
                    logging.info(f"Loaded existing results for {biography_title} with {model_name}")
                    biography_results["Models"].append(model_result)
                    continue
            except Exception as e:
                logging.warning(f"Could not load existing result, will reprocess: {e}")
        
        logging.info(f"Using model: {model_name}")
        
        try:
            # Initialize the language model
            llm = ChatOllama(model=model_name)
            
            # Split the documents into chunks
            chunks = split_documents(biography_docs)

            # Create the vector database
            vector_db = create_vector_db(chunks, biography_title, model_name)

            # Create the retriever
            retriever = create_retriever(vector_db, llm)

            # Create the chain
            chain = create_chain(retriever, llm)

            # Define the question
            question = f"Identify exactly 5 critical milestones or key developments from the biography of: {biography_title}?"

            # Get the response
            response = chain.invoke(input=question)
            logging.info(f"Response for {biography_title}: {response}")

            # Extract critical events and explanations
            critical_events, top_event_explanation = extract_critical_biographical_events(response, model_name)
            
            # Determine model personality using LLM analysis
            model_personality = llm_determined_personality(response, critical_events, model_name, biography_title)
            logging.info(f"Determined model personality: {model_personality}")
            
            # Make sure critical_events has at least one item
            if not critical_events:
                critical_events = ["No critical milestones identified"]
            
            # Prepare model result
            model_result = {
                "Model": model_name,
                "Model_Personality": model_personality,
                "Top_Critical_Event": critical_events[0],
                "Why_Critical": top_event_explanation,
                "Other_Key_Events": critical_events[1:5] if len(critical_events) > 1 else [],
                "Full_Response": response
            }
            
            biography_results["Models"].append(model_result)
            
            # Save intermediate results to avoid losing progress
            sanitized_model_name = model_name.replace(':', '_')
            with open(os.path.join(RESULTS_DIR, f"{biography_filename}_{sanitized_model_name}.json"), "w") as f:
                json.dump(model_result, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error processing {biography_title} with {model_name}: {str(e)}")
            error_result = {
                "Model": model_name,
                "Model_Personality": "Error",
                "Top_Critical_Event": "Processing failed",
                "Why_Critical": f"Error: {str(e)}",
                "Other_Key_Events": [],
                "Full_Response": f"Error during processing: {str(e)}"
            }
            biography_results["Models"].append(error_result)
    
    # Save the full biography results
    with open(os.path.join(RESULTS_DIR, f"{biography_filename}_analysis.json"), "w") as f:
        json.dump(biography_results, f, indent=4)
    
    # Also update the full output file
    combined_dataset_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    if os.path.exists(combined_dataset_path):
        try:
            with open(combined_dataset_path, 'r') as f:
                combined_dataset = json.load(f)
                
            # Check if this biography is already in the dataset
            biography_exists = False
            for i, biography in enumerate(combined_dataset):
                if biography["Biography"] == biography_title:
                    combined_dataset[i] = biography_results
                    biography_exists = True
                    break
            
            if not biography_exists:
                combined_dataset.append(biography_results)
                
            with open(combined_dataset_path, 'w') as f:
                json.dump(combined_dataset, f, indent=4)
        except:
            # If there's an error reading or updating, just overwrite with a new dataset
            with open(combined_dataset_path, 'w') as f:
                json.dump([biography_results], f, indent=4)
    else:
        # If the file doesn't exist, create it
        with open(combined_dataset_path, 'w') as f:
            json.dump([biography_results], f, indent=4)
    
    return biography_results

def analyze_all_biographies():
    """Analyze all biographies in the directory."""
    ensure_directories()
    
    # Load all biographies
    documents, biography_titles = load_biography_datasets()
    
    if not documents:
        logging.error("No biography documents to analyze")
        return None
    
    # Analyze each biography
    all_results = []
    for i, (doc, biography_title) in enumerate(zip(documents, biography_titles)):
        logging.info(f"Processing biography {i+1}/{len(documents)}: {biography_title}")
        biography_result = analyze_biography([doc], biography_title)
        all_results.append(biography_result)
    
    # Generate reports
    generate_reports()
    
    return all_results

def analyze_specific_biography(biography_name):
    """Analyze a specific biography by name."""
    ensure_directories()
    
    # Load the specific biography
    documents, biography_titles = load_biography_datasets(biography_name=biography_name)
    
    if not documents:
        logging.error(f"Biography '{biography_name}' not found or failed to load")
        return None
    
    # Analyze the biography
    biography_result = analyze_biography(documents, biography_titles[0])
    
    # Generate reports
    generate_reports()
    
    return biography_result

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
    for biography in all_results:
        biography_title = biography["Biography"]
        for model_result in biography["Models"]:
            row = {
                "Biography": biography_title,
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
    df.to_csv(os.path.join(RESULTS_DIR, "biographies_analysis_summary.csv"), index=False)
    
    # Generate model personality statistics
    personality_stats = df.groupby(['Model', 'Model_Personality']).size().reset_index(name='Count')
    personality_stats.to_csv(os.path.join(RESULTS_DIR, "biography_model_personality_stats.csv"), index=False)
    
    # Generate HTML report
    generate_html_report(df)
    
    # Generate model personality comparison table
    create_model_personality_table()
    
    logging.info("Reports generated successfully")

def generate_html_report(df):
    """Generate an HTML report with visualizations."""
    # Create a basic HTML report with tables and descriptions
    biography_count = df['Biography'].nunique()
    model_count = df['Model'].nunique()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Biographies Critical Milestones Analysis</title>
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
        <h1>Biographies Critical Milestones Analysis</h1>
        <p>Analyzed {biography_count} biographies using {model_count} different models with LLM-based personality detection.</p>
        
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
        
        <h2>Sample Biographies Analysis</h2>
    """
    
    # Add sample biography analyses
    for biography in df['Biography'].unique()[:5]:  # Show first 5 biographies
        biography_data = df[df['Biography'] == biography]
        html_content += f"""
        <h3>{biography}</h3>
        """
        
        for _, row in biography_data.iterrows():
            html_content += f"""
            <div class="model-card">
                <div class="model-header">
                    <strong>Model:</strong> {row['Model']} ({row['Model_Personality']})
                </div>
                <p><strong>Top Critical Milestone:</strong> {row['Top_Critical_Event']}</p>
                <p><strong>Why Critical:</strong> {row['Why_Critical']}</p>
                <p><strong>Other Key Milestones:</strong></p>
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
    with open(os.path.join(RESULTS_DIR, "biography_report.html"), "w") as f:
        f.write(html_content)

def create_model_personality_table():
    """Create a comparison table of model personalities."""
    try:
        with open(os.path.join(RESULTS_DIR, OUTPUT_FILE), 'r') as f:
            all_results = json.load(f)
            
        # Create a DataFrame to store the comparison
        comparison_data = []
        
        for biography in all_results:
            biography_title = biography["Biography"]
            
            # Extract model personalities for each model
            model_personalities = {}
            model_top_events = {}
            
            for model_result in biography["Models"]:
                model_name = model_result["Model"]
                model_personalities[model_name] = model_result["Model_Personality"]
                model_top_events[model_name] = model_result["Top_Critical_Event"]
            
            # Add to comparison data
            comparison_data.append({
                "Biography": biography_title,
                **{f"{model}_Personality": model_personalities.get(model, "N/A") for model in MODELS},
                **{f"{model}_TopEvent": model_top_events.get(model, "N/A") for model in MODELS}
            })
        
        # Create the DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_df.to_csv(os.path.join(RESULTS_DIR, "biography_model_personality_comparison.csv"), index=False)
        
        # Create an HTML version of the table
        html_table = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Personality Comparison for Biographies</title>
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
            <h1>Model Personality Comparison Across Biographies</h1>
            <p>Using LLM-based personality detection</p>
            <table>
                <tr>
                    <th>Biography</th>
        """
        
        for model in MODELS:
            html_table += f"""
                    <th>{model} Personality</th>
                    <th>{model} Top Milestone</th>
            """
        
        html_table += """
                </tr>
        """
        
        for _, row in comparison_df.iterrows():
            html_table += f"""
                <tr>
                    <td>{row['Biography']}</td>
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
        with open(os.path.join(RESULTS_DIR, "biography_model_personality_comparison.html"), "w") as f:
            f.write(html_table)
        
        return comparison_df
    except Exception as e:
        logging.error(f"Error creating model personality table: {str(e)}")
        return None

def main():
    """Main function to run the analysis."""
    # Example usage - analyze all biographies
    all_results = analyze_all_biographies()

if __name__ == "__main__":
    main()
