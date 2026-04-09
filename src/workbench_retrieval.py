import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)

def get_auth_token(session, email, password):
    """
    Authenticates with the API using a session object to retrieve a token.

    Args:
        session (requests.Session): The session object for making HTTP requests.
        email (str): The user's email address for authentication.
        password (str): The user's password for authentication.

    Returns:
        str: The authentication token if successful, None otherwise.
    """
    login_url = os.getenv("LOGIN_URL")
    login_payload = {"email": email, "password": password}

    try:
        with session.post(f'{login_url}/user/login', json=login_payload, timeout=10) as response:
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("success"):
                print("Authentication successful.")
                return response_data.get("payLoad", {}).get("authToken")
            else:
                print(f"Authentication failed: {response_data.get('message')}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during authentication: {e}")
        return None

def get_cdss_data_chunk(session, auth_token, patient_ids_chunk):
    """
    Fetches CDSS data for a single chunk of patient IDs.

    Args:
        session (requests.Session): The session object for making HTTP requests.
        auth_token (str): The authentication token.
        patient_ids_chunk (list): A list (chunk) of patient ECRF IDs.

    Returns:
        list: A list of patient data dictionaries if successful, None otherwise.
    """
    cdss_url = f"{os.getenv('LOGIN_URL')}/integration/get_cdss_data"
    headers = {'Authorization': f'Bearer {auth_token}'}

    try:
        with session.post(cdss_url, headers=headers, json=patient_ids_chunk, timeout=1200) as response:
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("success"):
                return response_data.get("payLoad", [])
            else:
                print(f"Failed to fetch data for chunk. Message: {response_data.get('message')}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching a CDSS data chunk: {e}")
        return None

def parse_to_parquet_and_save_ids(data, output_dir):
    """
    Parses the complex JSON response and saves SNV, CNA, and Fusion data into separate Parquet files
    in the specified output directory. Also saves the list of successfully retrieved patient IDs.

    Args:
        data (list): The list of patient data from the API response.
        output_dir (str): The directory to save the Parquet files and retrieved IDs list.
    """
    snv_rows, cna_rows, fusion_rows = [], [], []
    retrieved_ids = set()

    for patient in data:
        patient_id = patient.get('patientID')
        if patient_id:
            retrieved_ids.add(patient_id)
        for snv in patient.get('snvDataList', []):
            snv_rows.append({'patientID': patient_id, **snv})
        for cna in patient.get('cnaDataList', []):
            cna_rows.append({'patientID': patient_id, **cna})
        for fusion in patient.get('fusionDataList', []):
            fusion_rows.append({'patientID': patient_id, **fusion})

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        files_created = []
        
        if snv_rows:
            snv_df = pd.DataFrame(snv_rows)
            # Create the "Impact" column as concatenation of geneName and clinicalSignificanceOfTheVariant
            if 'geneName' in snv_df.columns and 'clinicalSignificanceOfTheVariant' in snv_df.columns:
                snv_df['Impact'] = snv_df['geneName'].astype(str) + "_" + snv_df['clinicalSignificanceOfTheVariant'].astype(str)
            else:
                snv_df['Impact'] = ""
            snv_file = os.path.join(output_dir, 'snv_cdss_input.parquet')
            snv_df.to_parquet(snv_file, index=False)
            files_created.append(snv_file)
            print(f"Created SNV file with {len(snv_df)} records")
        else:
            print("Warning: No SNV data found")
            
        if cna_rows:
            cna_df = pd.DataFrame(cna_rows)
            cna_df['Impact'] = cna_df['geneName'].astype(str) + "_" + cna_df['clinicalSignificanceOfTheVariant'].astype(str)
            cna_file = os.path.join(output_dir, 'cnv_cdss_input.parquet')
            cna_df.to_parquet(cna_file, index=False)
            files_created.append(cna_file)
            print(f"Created CNA file with {len(cna_df)} records")
        else:
            print("Warning: No CNA data found")
            
        if fusion_rows:
            fusion_df = pd.DataFrame(fusion_rows)
            fusion_df['Impact'] = fusion_df['clinicalSignificanceOfTheVariant'].astype(str)
            fusion_file = os.path.join(output_dir, 'fusion_cdss_input.parquet')
            fusion_df.to_parquet(fusion_file, index=False)
            files_created.append(fusion_file)
            print(f"Created Fusion file with {len(fusion_df)} records")
        else:
            print("Warning: No Fusion data found")

        # Save retrieved patient IDs
        if retrieved_ids:
            retrieved_file = os.path.join(output_dir, 'retrieved_list.txt')
            with open(retrieved_file, 'w') as f:
                for pid in sorted(retrieved_ids):
                    f.write(f"{pid}\n")
            files_created.append(retrieved_file)
            print(f"Created retrieved_list.txt with {len(retrieved_ids)} patient IDs")
        else:
            print("ERROR: No patient IDs were retrieved! This will cause the clinical data extraction to fail.")
            
        if files_created:
            print(f"\nData successfully parsed and saved to '{output_dir}' (as Parquet files).")
            print(f"Files created: {', '.join([os.path.basename(f) for f in files_created])}")
        else:
            print("\nERROR: No files were created. Check the API response data.")
    except Exception as e:
        print(f"An error occurred while writing Parquet files or retrieved list: {e}")

def load_ids_from_file(filename="samples.txt"):
    """
    Loads a list of sample IDs from a text file.

    Args:
        filename (str): The name of the file containing the IDs, one per line.

    Returns:
        list: A list of sample IDs, or an empty list if the file is not found or is empty.
    """
    try:
        with open(filename, 'r') as f:
            # Read each line, strip whitespace, and filter out any empty lines
            ids = [line.strip() for line in f if line.strip()]
        if not ids:
            print(f"Warning: '{filename}' is empty or contains no valid IDs.")
        return ids
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please create this file in the same directory as the script, with one sample ID per line.")
        return []

if __name__ == "__main__":
    load_dotenv() # Load environment variables from .env file

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Retrieve genomic data from CDSS API')
    parser.add_argument('--samples', type=str, required=True,
                       help='Path to samples file containing patient IDs.')
    parser.add_argument('--output_dir', type=str, default='genomic_data_parquet',
                       help='Directory to save the output Parquet files (default: genomic_data_parquet)')
    args = parser.parse_args()

    # --- Configuration ---
    user_email = os.getenv("USER_EMAIL")
    user_password = os.getenv("USER_PASSWORD")
    id_filename = args.samples
    output_directory = args.output_dir

    # Load ECRF IDs from the specified file
    print(f"Loading sample IDs from '{id_filename}'...")
    ecrf_ids = load_ids_from_file(id_filename)

    # Proceed only if IDs were successfully loaded from the file
    if ecrf_ids:
        # --- Main Execution ---
        all_patient_data = []
        with requests.Session() as session:
            print("Attempting to authenticate...")
            token = get_auth_token(session, user_email, user_password)

            if token:
                print("Fetching CDSS data concurrently...")
                # Chunk the patient IDs into lists of 500 (API limit)
                chunk_size = 200
                id_chunks = [ecrf_ids[i:i + chunk_size] for i in range(0, len(ecrf_ids), chunk_size)]

                with ThreadPoolExecutor(max_workers=10) as executor:
                    # Submit all chunks to the executor
                    future_to_chunk = {executor.submit(get_cdss_data_chunk, session, token, chunk): chunk for chunk in id_chunks}

                    for future in as_completed(future_to_chunk):
                        result = future.result()
                        if result:
                            all_patient_data.extend(result)

                if all_patient_data:
                    print(f"\nSuccessfully fetched data for {len(all_patient_data)} records.")
                    parse_to_parquet_and_save_ids(all_patient_data, output_directory)
                    
                    # Verify that critical files were created in the specified output directory
                    retrieved_list_path = os.path.join(output_directory, 'retrieved_list.txt')
                    if not os.path.exists(retrieved_list_path):
                        print(f"\nERROR: Critical file {retrieved_list_path} was not created!")
                        exit(1)
                    else:
                        print(f"\nGenomic data retrieval completed successfully.")
                        print(f"Critical file {retrieved_list_path} confirmed to exist.")
                else:
                    print("\nNo data was fetched from the CDSS API.")
                    print("This could be due to:")
                    print("- Invalid patient IDs in the samples file")
                    print("- Network connectivity issues")
                    print("- API authentication problems")
                    print("- API server issues")
                    exit(1)
            else:
                print("\nCould not retrieve authentication token. Exiting.")
                exit(1)
    else:
        print("No sample IDs loaded. Exiting.")
        exit(1)
