import sys
import os
import logging
import pandas as pd
import requests
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import tempfile
import shutil
import argparse

load_dotenv(override=True)

wd = os.getcwd()

## Patient Data Exporter
class PatientDataExporter:
    def __init__(self, pathforjsons, samples, api_url, resume=False, max_workers=20):
        self.pathforjsons = pathforjsons
        self.samples = samples
        self.api_url = api_url
        self.resume = resume
        self.max_workers = max_workers
        self.session = self._create_session()
        self.token = self._load_token()
        self._setup_logging()
        self._create_output_directory()

    def _create_session(self):
        """Create a session with connection pooling and retry logic"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _load_token(self):
        ecrf_login_url = os.getenv("ECRF_LOGIN_URL", "https://www.v2.api.ecrf.4basecare.co.in/user/login")
        body = {
            "email": os.getenv("ECRF_EMAIL"),
            "password": os.getenv("ECRF_PASSWORD")
        }
        
        if not body["email"] or not body["password"]:
            raise ValueError("ECRF_EMAIL and ECRF_PASSWORD must be set in environment variables")
        
        try:
            response = self.session.post(ecrf_login_url, json=body, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            
            if 'payLoad' in response_data and 'authToken' in response_data['payLoad']:
                return response_data['payLoad']['authToken']
            else:
                raise ValueError(f"Authentication token not found in response. Response: {response_data}")
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Authentication request failed: {e}")

    def _setup_logging(self):
        """
        Sets up logging to write to a log file in the specified directory.
        """
        os.makedirs(self.pathforjsons, exist_ok=True)
        log_file = os.path.join(self.pathforjsons, 'data_export.log')
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        logging.info("Logging setup complete.")

    def _create_output_directory(self):
        # No longer uses os.chdir, just ensures the directory exists.
        os.makedirs(self.pathforjsons, exist_ok=True)

    def _fetch_patient_data(self, patient_id):
        """Fetch data for a single patient"""
        url = f"{self.api_url}{patient_id}"
        headers = {'Authorization': f'Bearer {self.token}'}
        
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return patient_id, response.json()
            else:
                logging.error(f"Error fetching data for {patient_id}. Status code: {response.status_code}")
                return patient_id, None
        except Exception as e:
            logging.error(f"Exception fetching data for {patient_id}: {e}")
            return patient_id, None

    def _export_patient_data(self, patient_id, patient_data):
        """Export patient data to a JSON file in the designated directory."""
        output_file = os.path.join(self.pathforjsons, f"{patient_id}_data.json")
        with open(output_file, 'w') as json_file:
            json.dump(patient_data, json_file, indent=2)
        logging.info(f"Data for {patient_id} exported successfully to {output_file}")

    def fetch_and_export_data(self, patient_id):
        """Fetch and export data for a single patient, using full paths."""
        output_file = os.path.join(self.pathforjsons, f"{patient_id}_data.json")
        if self.resume and os.path.exists(output_file):
            logging.info(f"Skipping {patient_id}, file already exists.")
            return patient_id, None
        
        patient_id, patient_data = self._fetch_patient_data(patient_id)
        if patient_data:
            self._export_patient_data(patient_id, patient_data)
        return patient_id, patient_data

    def process_patients(self):
        """Process all patients concurrently"""
        try:
            pat_df = pd.read_csv(self.samples, header=None)
            patient_ids = pat_df[0].astype(str).str.strip().tolist()
        except FileNotFoundError:
            logging.error(f"Samples file not found at {self.samples}")
            return
        
        patient_ids = [pid for pid in patient_ids if pid]
        print(f"Processing {len(patient_ids)} patients with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_patient = {
                executor.submit(self.fetch_and_export_data, patient_id): patient_id 
                for patient_id in patient_ids
            }
            
            completed = 0
            for future in as_completed(future_to_patient):
                patient_id = future_to_patient[future]
                try:
                    future.result()
                    completed += 1
                    print(f'Processed {completed}/{len(patient_ids)}: {patient_id}', end="\r")
                except Exception as e:
                    logging.error(f"Exception processing {patient_id}: {e}")
        
        print(f"\nCompleted processing {len(patient_ids)} patients.")

    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()


# Optimized Cancer Details Processing
def process_cancer_details_optimized(json_directory, output_dir, info="cancerDetails", output_pq_name="cancerDetails.parquet", resume=False):
    """
    Optimized version that processes JSON files and saves parquet to a specified output directory.
    """
    json_list = [file for file in os.listdir(json_directory) if file.endswith('.json')]
    
    info_directory = os.path.join(json_directory, info)
    os.makedirs(info_directory, exist_ok=True)
    
    all_data = []
    
    for json_file in json_list:
        input_path = os.path.join(json_directory, json_file)
        
        try:
            df = pd.read_json(input_path)
            df.drop(['success', 'message'], axis=1, inplace=True, errors='ignore')
            df = df.T
            
            if info in df.columns:
                extracted_data = df[info].iloc[0] if len(df) > 0 else {}
                if extracted_data:
                    normalized_data = pd.json_normalize(extracted_data, sep='_')
                    normalized_data['patientID'] = json_file.replace('_data.json', '')
                    all_data.append(normalized_data)
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if all_data:
        # Filter out empty dataframes to prevent a FutureWarning from pd.concat
        all_data = [df for df in all_data if not df.empty]
        if not all_data:
            return # Exit if all dataframes were empty

        final_df = pd.concat(all_data, ignore_index=True)
        # Save the final parquet file to the specified output directory
        output_path = os.path.join(output_dir, f"{info}_{output_pq_name}")
        final_df.to_parquet(output_path)
        # Quieter output: The following print statement has been removed.
        # print(f"\nIntermediate data for '{info}' saved to {output_path}")
    # else:
        # Quieter output: The following print statement has been removed.
        # print(f"\nNo data found for {info}")

def process_data(input_dir, output_dir, resume=False, max_workers=10):  
    """
    Main data processing function that uses a temporary directory for intermediate files.
    """
    # Ensure the output directory exists before creating a temp directory inside it
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=output_dir, prefix="ecrf_json_")
    
    pathforjsons = os.path.join(temp_dir, 'json_response')
    samples = os.path.join(input_dir, "retrieved_list.txt") # Read from the input directory
    api_url = "https://www.v2.api.ecrf.4basecare.co.in/integration/getExternalApiResponseByPatientId/"
    
    try:
        if not resume:
            exporter = PatientDataExporter(pathforjsons, samples, api_url, resume=resume, max_workers=max_workers)
            exporter.process_patients()
        
        # Process details, saving intermediate parquet to the temp directory
        process_cancer_details_optimized(pathforjsons, temp_dir, info="cancerDetails", output_pq_name="cancerDetails.parquet", resume=resume)
        process_cancer_details_optimized(pathforjsons, temp_dir, info="patientInfo", output_pq_name="patientDetails.parquet", resume=resume)
        process_cancer_details_optimized(pathforjsons, temp_dir, info="medicalInfo", output_pq_name="medicalDetails.parquet", resume=resume)
        
        # Merge the data from the temp directory
        try:
            cancer_df_path = os.path.join(temp_dir, "cancerDetails_cancerDetails.parquet")
            patient_df_path = os.path.join(temp_dir, "patientInfo_patientDetails.parquet")
            medical_df_path = os.path.join(temp_dir, "medicalInfo_medicalDetails.parquet")
            
            cancer_df = pd.read_parquet(cancer_df_path)
            patient_df = pd.read_parquet(patient_df_path)
            medical_df = pd.read_parquet(medical_df_path)
            
            clinical_df = cancer_df.merge(patient_df, on="patientID", how="inner")
            final_df = clinical_df.merge(medical_df, on="patientID", how="inner")
            
            # Save the final output to the designated output directory
            final_output_path = os.path.join(output_dir, "clinical_Details.parquet")
            final_df.to_parquet(final_output_path, index=False)
            print(f"Final merged data saved to: {final_output_path}")
            
        except FileNotFoundError as e:
            print(f"\nError: Could not find intermediate file for merging: {e}", file=sys.stderr)
        except Exception as e:
            print(f"\nAn error occurred during the final merge: {e}", file=sys.stderr)
    
    finally:
        # Shut down logging to release file locks before cleanup
        logging.shutdown()
        # Clean up the temporary directory for JSONs, handling potential errors
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"Warning: Could not remove temporary directory {temp_dir}. Reason: {e}", file=sys.stderr)


def main():
    """
    Main execution block that sets up and cleans up a temporary directory.
    """
    parser = argparse.ArgumentParser(description='Extract and process clinical data.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input files like retrieved_list.txt')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the final clinical_Details.parquet file.')
    parser.add_argument('--resume', action='store_true', help='Resume processing from existing files')
    parser.add_argument('--workers', type=int, default=20, help='Number of concurrent workers (default: 20)')
    args = parser.parse_args()
    
    # The tempfile management is now handled within process_data
    process_data(args.input_dir, args.output_dir, args.resume, args.workers)

if __name__ == "__main__":
    main()