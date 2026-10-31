import os
import time
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load Environment Variables (.env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found.")

genai.configure(api_key=GEMINI_API_KEY)


model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    generation_config={"response_mime_type": "application/json"}
)

# 2. Configuration Parameters
INPUT_FILE = "datasets/comments_labeled.csv"
OUTPUT_FILE = "datasets/comments_labeled_paraphrased.csv"
BATCH_SIZE = 50       # Number of comments per API request batch
DELAY_BETWEEN_BATCHES = 2  # Delay in seconds to respect rate limits

# 3. Batch Paraphrasing Function (Prompt is in Indonesian)
def paraphrase_batch(batch_items):
    """
    batch_items: list of dicts [{'id': int, 'text': str}, ...]
    returns: dict {id: paraphrased_text}
    """
    prompt = f"""
    Ubah susunan kata (paraphrase) dari daftar komentar berikut ke dalam bahasa gaul/santai netizen Indonesia.

    Aturan Ketat:
    1. Hapus semua mention username (@username), nama orang, atau entitas pribadi (PII).
    2. Jaga agar MAKNA UTAMA dan TONE SENTIMEN (Pro/Kontra) tetap 100% sama dengan aslinya.
    3. Buat susunan kata dan struktur kalimatnya berbeda agar tidak bisa dicari secara eksak (exact match) di mesin pencari.
    4. Kembalikan hasil HANYA dalam format JSON Array of Objects dengan skema:
       [
         {{
           "id": <id_komentar>,
           "paraphrased_text": "<hasil_parafrase>"
         }}
       ]

    Daftar Komentar Asli:
    {json.dumps(batch_items, ensure_ascii=False, indent=2)}
    """
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            result_json = json.loads(response.text.strip())
            
            # Convert JSON array response to lookup dictionary
            paraphrased_dict = {}
            for item in result_json:
                paraphrased_dict[item["id"]] = item["paraphrased_text"]
            return paraphrased_dict
            
        except Exception as e:
            print(f"   [!] Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(4)
            else:
                print("   [!] Batch processing failed. Falling back to original texts for this batch.")
                # Fallback: return original text mapped by ID
                return {item["id"]: item["text"] for item in batch_items}

# 4. Main Processing Pipeline
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' does not exist.")
        return

    print(f"Reading dataset from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    if 'text' not in df.columns:
        raise ValueError("The 'text' column is missing from the dataset.")

    total_comments = len(df)
    print(f"Successfully loaded {total_comments} comments.")
    print(f"Starting batch paraphrasing process (Batch Size: {BATCH_SIZE})...\n")

    # Prepare data into ID-indexed batch structures
    items = [{"id": idx, "text": str(row['text'])} for idx, row in df.iterrows()]
    
    paraphrased_results = {}
    total_batches = (total_comments + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total_comments, BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        print(f"Processing Batch [{current_batch_num}/{total_batches}] ({len(batch)} items)...")
        
        batch_output = paraphrase_batch(batch)
        paraphrased_results.update(batch_output)
        
        print(f"   -> Batch [{current_batch_num}/{total_batches}] completed successfully.")
        time.sleep(DELAY_BETWEEN_BATCHES)

    # Map the paraphrased output back to DataFrame
    df['text'] = [paraphrased_results.get(idx, row['text']) for idx, row in df.iterrows()]

    # Save output dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[DONE] Paraphrasing pipeline finished!")
    print(f"Paraphrased dataset successfully saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()