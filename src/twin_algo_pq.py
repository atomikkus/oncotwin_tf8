import os
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse


## Basic Preprocessing
def preprocess_dataframe(df):
    df.columns = df.columns.astype(str).str.strip().str.replace('\n', '') 
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip().str.replace('\n', '')
        df[column] = df[column].replace(['nan', 'NA'], [np.nan, np.nan])

    return df

## Creating Patient Profiles
def create_patient_profiles(snv_df, cnv_df, fusion_df, clinical_df, 
                            snv_subset, cnv_subset, fusion_subset, clinical_subset, output_path):

    # Define subsets dynamically
    subsets = {
        'snv': {'df': snv_df, 'cols': snv_subset},
        'cnv': {'df': cnv_df, 'cols': cnv_subset},
        'fusion': {'df': fusion_df, 'cols': fusion_subset},
        'clinical': {'df': clinical_df, 'cols': clinical_subset},
    }

    os.makedirs(output_path, exist_ok=True)

    # Fields with prefixes
    all_fields = set()
    for subset_name, subset_info in subsets.items():
        prefixed_fields = [f"{subset_name}_{col}" for col in subset_info['cols'] if col != 'patientID']
        all_fields.update(prefixed_fields)

    patient_profiles = defaultdict(lambda: defaultdict(set))

    for subset_name, subset_info in subsets.items():
        df = subset_info['df']
        cols = subset_info['cols']

        # Relevant columns are selected
        filtered_df = df[cols]

        for _, row in filtered_df.iterrows():
            patient_id = row['patientID']
            data_entry = row.drop('patientID').to_dict()

            for key, value in data_entry.items():
                if pd.notna(value):  # Ignore NaN values
                    prefixed_key = f"{subset_name}_{key}"
                    patient_profiles[patient_id][prefixed_key].add(value)

    # Ensure all fields are present in every patient profile
    for patient_id, profile in patient_profiles.items():
        for field in all_fields:
            if field not in profile:
                profile[field] = set()  # Keep it empty instead of NaN
    # Convert sets to lists for JSON serialization (create a new dict)
    patient_profiles_list = {pid: {key: list(values) for key, values in profile.items()} for pid, profile in patient_profiles.items()}

    # Write each patient profile to a JSON file
    for patient_id, profile in patient_profiles_list.items():
        filename = os.path.join(output_path, f"{patient_id}.json")
        with open(filename, 'w') as json_file:
            json.dump(profile, json_file, indent=4)
    return patient_profiles_list
            
                        
def jaccard_similarity(set1, set2):
    # Convert set elements to integers if all elements are numeric
    try:
        num_set1 = {int(x) for x in set1}
        num_set2 = {int(x) for x in set2}
        
        # Compute numeric similarity using 1 - (difference / max)
        max_val = max(max(num_set1, default=0), max(num_set2, default=0))
        return 1 - (abs(sum(num_set1) - sum(num_set2)) / (max_val + 1e-9))  # Normalize
    except ValueError:
        pass  # If conversion fails, proceed with Jaccard similarity

    # Standard Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def calculate_similarity(patient_profiles, top_n, weights):
    # Get patient IDs and their corresponding profiles
    patient_ids = list(patient_profiles.keys())
    matches = {}

    max_scores = {}
    for patient_id in patient_ids:
        max_score = 0
        for key in patient_profiles[patient_id]:
            set1 = set(patient_profiles[patient_id][key])
            if set1:  # Calculate self-similarity for non-empty fields
                weight = weights.get(key, 1.0)
                max_score += weight * jaccard_similarity(set1, set1)
        max_scores[patient_id] = max_score

    # Calculate normalized similarity for each patient pair
    for i, patient_id in enumerate(tqdm(patient_ids, desc='Matching patients')):
        similarity_scores = []
        for j, other_patient_id in enumerate(patient_ids):
            if i != j:
                score = 0
                valid_fields = 0  
                
                # Calculate similarity for each profile attribute
                for key in patient_profiles[patient_id]:
                    set1 = set(patient_profiles[patient_id][key])
                    set2 = set(patient_profiles[other_patient_id][key])

                    # for fusions
                    if key == "fusion_gene3" and "fusion_gene5" in patient_profiles[other_patient_id]:
                        set2_union = set2.union(set(patient_profiles[other_patient_id]["fusion_gene5"]))
                        if set1:
                            weight = weights.get(key, 1.0)
                            score += weight * jaccard_similarity(set1, set2_union)
                            valid_fields += 1
                    elif key == "fusion_gene5" and "fusion_gene3" in patient_profiles[other_patient_id]:
                        set2_union = set2.union(set(patient_profiles[other_patient_id]["fusion_gene3"]))
                        if set1:
                            weight = weights.get(key, 1.0)
                            score += weight * jaccard_similarity(set1, set2_union)
                            valid_fields += 1
                    else:
                        if set1:
                            weight = weights.get(key, 1.0)
                            score += weight * jaccard_similarity(set1, set2)
                            valid_fields += 1

                # Normalize the score by dividing by the max score for the patient
                if valid_fields > 0 and max_scores[patient_id] > 0:
                    normalized_score = score / max_scores[patient_id]
                    similarity_scores.append((other_patient_id, normalized_score))

        # Sort similarity scores in descending order and select top N matches
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        matches[patient_id] = similarity_scores[:top_n]

    return matches

def matches_to_dataframe(matches, patient_profiles):
    # Create a list of rows to store query and similar patient information
    rows = []

    for query_patient, similar_patients in matches.items():
        query_data = patient_profiles.get(query_patient, {})

        for similar_patient, score in similar_patients:
            similar_data = patient_profiles.get(similar_patient, {})

            # Flatten query and similar patient profiles into a single dictionary
            row = {
                "Query": query_patient.replace(".json", ""),
                "Similar": similar_patient.replace(".json", ""),
                "Score": score,
            }
            # Add fields from the query patient
            for field, values in query_data.items():
                row[f"Query_{field}"] = ", ".join(map(str, values)) if values else ""

            # Add fields from the similar patient
            for field, values in similar_data.items():
                row[f"Similar_{field}"] = ", ".join(map(str, values)) if values else ""

            rows.append(row)

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(rows)
    return df

def main():
    parser = argparse.ArgumentParser(description='Twin matching algorithm with optional single-patient mode')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing all input parquet and config files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the final output files.')
    parser.add_argument('--single', type=str, help='Patient ID to calculate matches for (only this patient)')
    parser.add_argument('--json_output', action='store_true', help='Save results as JSON instead of Excel')
    parser.add_argument('--weights', type=str, help='Path to custom weights.json file.')
    parser.add_argument('--subsets', type=str, help='Path to custom column_subsets.json file.')
    parser.add_argument('--doctor_id', type=int, help='Doctor ID to be included in the output.')
    args = parser.parse_args()

    # --- Load Configuration Files ---
    script_dir = os.path.dirname(os.path.realpath(__file__)) # Get directory of the script
    
    # Determine paths for config files, using arguments if provided, otherwise default to config directory
    subsets_path = args.subsets if args.subsets else os.path.join(os.path.dirname(script_dir), 'config', 'column_subsets.json')
    weights_path = args.weights if args.weights else os.path.join(os.path.dirname(script_dir), 'config', 'weights.json')

    
    try:
        with open(subsets_path, 'r') as f:
            subsets = json.load(f)
            snv_subset = subsets['snv_subset']
            cnv_subset = subsets['cnv_subset']
            fusion_subset = subsets['fusion_subset']
            clinical_subset = subsets['clinical_subset']
    except FileNotFoundError:
        print(f"Warning: Subsets file not found at '{subsets_path}'. Using default column subsets.")
        snv_subset = ['patientID', 'geneName', 'variantPDot', 'Impact']
        cnv_subset = ['patientID', 'geneName']
        fusion_subset = ['patientID', 'gene5', 'gene3']
        clinical_subset = ['patientID', 'cancerSite', 'gender', 'age']

    default_weights = { "clinical_cancerSite": 0.1, "snv_variantPDot": 2, "snv_Impact": 3, "cnv_geneName": 7.6, "fusion_gene3": 9.25, "snv_geneMarker": 0.0, "snv_geneName": 4, "fusion_gene5": 9.25, "clinical_gender": 0.2, "clinical_morphologyIdcCode": 0.1, "clinical_age": 0.2 }
    try:
        with open(weights_path, 'r') as f:
            weights_2 = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Weights file not found at '{weights_path}'. Using default weights.")
        weights_2 = default_weights

    # --- Load Data Files ---
    # Check which genomic files exist
    snv_file = os.path.join(args.input_dir, 'snv_cdss_input.parquet')
    cnv_file = os.path.join(args.input_dir, 'cnv_cdss_input.parquet')
    fusion_file = os.path.join(args.input_dir, 'fusion_cdss_input.parquet')
    clinical_file = os.path.join(args.input_dir, "clinical_Details.parquet")
    
    genomic_files = {
        'snv': snv_file,
        'cnv': cnv_file,
        'fusion': fusion_file
    }
    
    # Check if at least one genomic file exists
    existing_genomic_files = [name for name, path in genomic_files.items() if os.path.exists(path)]
    
    if not existing_genomic_files:
        print(f"Error: No genomic data files found. Please ensure at least one of the following files exists:")
        for name, path in genomic_files.items():
            print(f"  - {name}: {path}")
        sys.exit(1)
    
    # Load existing genomic files, create empty dataframes for missing ones
    try:
        if os.path.exists(snv_file):
            snv_df = preprocess_dataframe(pd.read_parquet(snv_file))
        else:
            print(f"Warning: SNV file not found at {snv_file}. Using empty dataframe.")
            snv_df = pd.DataFrame(columns=snv_subset)
        
        if os.path.exists(cnv_file):
            cnv_df = preprocess_dataframe(pd.read_parquet(cnv_file))
            if 'functionCnv' in cnv_df.columns:
                cnv_df['functionCnv'] = cnv_df['functionCnv'].apply(lambda x : x.lower())
        else:
            print(f"Warning: CNV file not found at {cnv_file}. Using empty dataframe.")
            cnv_df = pd.DataFrame(columns=cnv_subset)
        
        if os.path.exists(fusion_file):
            fusion_df = preprocess_dataframe(pd.read_parquet(fusion_file))
        else:
            print(f"Warning: Fusion file not found at {fusion_file}. Using empty dataframe.")
            fusion_df = pd.DataFrame(columns=fusion_subset)
        
        if os.path.exists(clinical_file):
            clinical_df = pd.read_parquet(clinical_file)
            clinical_df.drop_duplicates(subset="patientID", inplace=True)
        else:
            print(f"Warning: Clinical file not found at {clinical_file}. Using empty dataframe.")
            clinical_df = pd.DataFrame(columns=clinical_subset)
            
    except Exception as e:
        print(f"Error: Failed to load data files - {e}")
        sys.exit(1)

    # --- Main Logic ---
    profiles_path = os.path.join(args.input_dir, "patient_profiles")
    patient_profiles = create_patient_profiles(snv_df, cnv_df, fusion_df, clinical_df, snv_subset, cnv_subset, fusion_subset, clinical_subset, profiles_path)

    if args.single:
        # Only calculate matches for the specified patient
        query_id = args.single
        if query_id not in patient_profiles:
            print(f"Patient ID {query_id} not found in profiles.")
            exit(1)
        # Calculate similarity only for this patient
        matches = {}
        patient_ids = list(patient_profiles.keys())
        max_score = 0
        for key in patient_profiles[query_id]:
            set1 = set(patient_profiles[query_id][key])
            if set1:
                weight = weights_2.get(key, 1.0)
                max_score += weight * jaccard_similarity(set1, set1)
        similarity_scores = []
        for other_patient_id in patient_ids:
            if other_patient_id == query_id:
                continue
            score = 0
            valid_fields = 0
            for key in patient_profiles[query_id]:
                set1 = set(patient_profiles[query_id][key])
                set2 = set(patient_profiles[other_patient_id][key])
                # for fusions
                if key == "fusion_gene3" and "fusion_gene5" in patient_profiles[other_patient_id]:
                    set2_union = set2.union(set(patient_profiles[other_patient_id]["fusion_gene5"]))
                    if set1:
                        weight = weights_2.get(key, 1.0)
                        score += weight * jaccard_similarity(set1, set2_union)
                        valid_fields += 1
                elif key == "fusion_gene5" and "fusion_gene3" in patient_profiles[other_patient_id]:
                    set2_union = set2.union(set(patient_profiles[other_patient_id]["fusion_gene3"]))
                    if set1:
                        weight = weights_2.get(key, 1.0)
                        score += weight * jaccard_similarity(set1, set2_union)
                        valid_fields += 1
                else:
                    if set1:
                        weight = weights_2.get(key, 1.0)
                        score += weight * jaccard_similarity(set1, set2)
                        valid_fields += 1
            if valid_fields > 0 and max_score > 0:
                normalized_score = score / max_score
            else:
                normalized_score = 0
            similarity_scores.append((other_patient_id, normalized_score))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarity_scores[:20]
        # Output as DataFrame
        matches = {query_id: top_matches}
        match_df = matches_to_dataframe(matches, patient_profiles)
        # Add cancer match column
        match_df['c_match'] = match_df['Query_clinical_cancerSite'] == match_df['Similar_clinical_cancerSite']
        match_df['cancer_match'] = np.where(match_df['c_match'], "same", "other")

        # Add doctor_id to the DataFrame if provided
        if args.doctor_id is not None:
            match_df['doctor_id'] = args.doctor_id

        if args.json_output:
            # Save as JSON with simplified structure
            json_results = { "query_patient_id": query_id, "matches": [] }
            # Use df.to_dict to handle columns consistently
            for record in match_df.to_dict('records'):
                match_info = {
                    "similar_patient_id": record.get("Similar"),
                    "matching_percentage": record.get("Score"),
                    "c_match_status": record.get("cancer_match")
                }
                if 'doctor_id' in record:
                    match_info['doctor_id'] = record['doctor_id']
                json_results["matches"].append(match_info)
            output_filename = os.path.join(args.output_dir, f"matches_{query_id}.json")
            with open(output_filename, 'w') as f:
                json.dump(json_results, f, indent=4)
            print(f"Output for patient {query_id} saved to {output_filename}")
        else:
            output_filename = os.path.join(args.output_dir, f"matches_scoring_{query_id}.xlsx")
            match_df.to_excel(output_filename, index=False)
            print(f"Output for patient {query_id} saved to {output_filename}")
        exit(0)

    patient_ids = os.listdir(profiles_path)
    profiles = {patient_id: json.load(open(os.path.join(profiles_path, patient_id))) for patient_id in patient_ids}
    matches = calculate_similarity(profiles, top_n=20, weights=weights_2)
    match_df = matches_to_dataframe(matches, profiles)

    print(match_df.columns)
    # List of columns to check
    columns_to_check = [
        "Query_snv_geneName", 
        "Query_snv_variantPDot", 
        "Query_fusion_gene3", 
        "Query_cnv_geneName", 
        "Query_fusion_gene5",
        "Query_snv_Impact"
    ]

    # Convert string 'nan', 'NaN', 'NAN' etc. to actual NaN values
    match_df[columns_to_check] = match_df[columns_to_check].replace(
        to_replace=["nan", "NaN", "NAN", ""], 
        value=np.nan
    )
    match_df = match_df.dropna(axis=0, how="all", subset=columns_to_check)
    match_df = match_df[match_df["Score"] != 0]
    # Print DataFrame statistics only if match_df is a DataFrame
    if isinstance(match_df, pd.DataFrame):
        print(match_df['Score'].describe())
        ## Rename score column to similarity
        match_df = match_df.rename(columns={"Score": "Matching Perc"})
    else:
        print("Warning: match_df is not a DataFrame, skipping describe and rename.")

    # Add doctor_id to the DataFrame if provided
    if args.doctor_id is not None:
        match_df['doctor_id'] = args.doctor_id

    ## cancer match column
    match_df['c_match'] = match_df['Query_clinical_cancerSite'] == match_df['Similar_clinical_cancerSite']
    match_df['cancer_match'] = np.where(match_df['c_match'], "same", "other")

    output_filename = os.path.join(args.output_dir, "matches_scoring_consolidated.xlsx")
    match_df.to_excel(output_filename, index=False)
    print(f"Consolidated Excel matches saved to {output_filename}")

    if args.json_output:
        # Save consolidated results as JSON with minimal structure
        json_results = { "matches": [] }
        # Use df.to_dict to handle columns consistently
        for record in match_df.to_dict('records'):
            match_info = {
                "query_patient_id": record.get("Query"),
                "similar_patient_id": record.get("Similar"),
                "matching_percentage": record.get("Matching Perc"),
                "c_match_status": record.get("cancer_match")
            }
            if 'doctor_id' in record:
                match_info['doctor_id'] = record['doctor_id']
            json_results["matches"].append(match_info)
        output_filename = os.path.join(args.output_dir, "matches_consolidated.json")
        with open(output_filename, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"Consolidated JSON matches saved to {output_filename}")
    else:
        output_filename = os.path.join(args.output_dir, "matches_scoring_consolidated.xlsx")
        match_df.to_excel(output_filename, index=False)
        print(f"Consolidated Excel matches saved to {output_filename}")


if __name__ == "__main__":
    import sys
    main()


