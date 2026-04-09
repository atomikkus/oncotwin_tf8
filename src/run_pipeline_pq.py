import subprocess
import argparse
import sys
import os
import hashlib
import json
import tempfile
import shutil


def get_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    if not os.path.exists(filepath): return None
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_previous_run(samples_file, work_dir):
    """Check for a state file within the working directory."""
    state_file = os.path.join(work_dir, "pipeline_state.json")
    if not os.path.exists(state_file):
        return False, False, None
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        current_hash = get_file_hash(samples_file)
        if state.get('samples_hash') != current_hash:
            return False, False, state
        
        genomic_files_exist = all(os.path.exists(os.path.join(work_dir, f)) for f in [
            "snv_cdss_input.parquet", "cnv_cdss_input.parquet", 
            "fusion_cdss_input.parquet", "retrieved_list.txt"
        ])
        
        clinical_files_exist = os.path.exists(os.path.join(work_dir, "clinical_Details.parquet"))
        
        return genomic_files_exist, clinical_files_exist, state
        
    except (json.JSONDecodeError, FileNotFoundError):
        return False, False, None

def save_pipeline_state(samples_file, work_dir):
    """Save pipeline state into the working directory."""
    state = {
        'samples_hash': get_file_hash(samples_file),
        'samples_file': os.path.basename(samples_file),
        'timestamp': datetime.fromtimestamp(os.path.getmtime(samples_file)).isoformat()
    }
    with open(os.path.join(work_dir, "pipeline_state.json"), 'w') as f:
        json.dump(state, f, indent=2)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f'\n{description}...')
    print(f'Running: {" ".join(cmd)}')
    try:
        # Using sys.executable ensures we use the same python interpreter
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ)
        print(f'{description} completed successfully.')
        return True
    except subprocess.CalledProcessError as e:
        print(f'Error in {description}:')
        print(f'Return code: {e.returncode}')
        print(f'stdout: {e.stdout.strip()}')
        print(f'stderr: {e.stderr.strip()}')
        return False

def main():
    parser = argparse.ArgumentParser(description='Run the full onco-twin pipeline with Parquet files')
    
    parser.add_argument('--samples', type=str, required=True, help='Path to .txt file containing list of sample IDs')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save final results.')
    parser.add_argument('--keep_temp_dir', action='store_true', help='Keep the temporary directory after the run for debugging.')
    
    # Arguments for component scripts
    parser.add_argument('--single', type=str, help='Patient ID for matching (optional)')
    parser.add_argument('--json_output', action='store_true', help='Save results as JSON instead of Excel')
    parser.add_argument('--weights', type=str, help='Path to custom weights.json file for the matching algorithm.')
    parser.add_argument('--subsets', type=str, help='Path to custom column_subsets.json file for the matching algorithm.')
    parser.add_argument('--doctor_id', type=int, help='Doctor ID to be passed to the matching algorithm.')
    
    # Pipeline control
    parser.add_argument('--skip_genomic', action='store_true', help='Skip genomic data retrieval')
    parser.add_argument('--skip_clinical', action='store_true', help='Skip clinical data extraction')
    parser.add_argument('--skip_matching', action='store_true', help='Skip matching algorithm')
    parser.add_argument('--refresh', action='store_true', help='Force refresh of data')
    
    args = parser.parse_args()
    
    # Create a temporary directory for the entire pipeline run
    work_dir = tempfile.mkdtemp(prefix="oncotwin_run_")
    print(f"Pipeline working directory: {work_dir}")
    
    # Create the final output directory if it doesn't exist
    final_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Final output will be in: {final_output_dir}")

    try:
        can_skip_genomic, can_skip_clinical, _ = check_previous_run(args.samples, work_dir)
        
        skip_genomic = (can_skip_genomic and not args.refresh) or args.skip_genomic
        skip_clinical = (can_skip_clinical and not args.refresh) or args.skip_clinical

        # Step 1: Genomic data retrieval
        if not skip_genomic:
            cmd = [sys.executable, 'src/workbench_retrieval.py', '--samples', args.samples, '--output_dir', work_dir]
            if not run_command(cmd, "Genomic data retrieval"):
                print("Pipeline failed at genomic data retrieval.")
                sys.exit(1)
            save_pipeline_state(args.samples, work_dir)
        else:
            print("Skipping genomic data retrieval.")
        
        # Step 2: Clinical data extraction
        if not skip_clinical:
            # Note: ecrf_extract_pq.py now reads from and writes to the working directory
            cmd = [sys.executable, 'src/ecrf_extract_pq.py', '--input_dir', work_dir, '--output_dir', work_dir]
            if not run_command(cmd, "Clinical data extraction"):
                print("Pipeline failed at clinical data extraction.")
                sys.exit(1)
        else:
            print("Skipping clinical data extraction.")
        
        # Step 3: Matching algorithm
        if not args.skip_matching:
            # The matching script reads all inputs from work_dir and writes final output to final_output_dir
            cmd = [sys.executable, 'src/twin_algo_pq.py', '--input_dir', work_dir, '--output_dir', final_output_dir]
            if args.single:
                cmd.extend(['--single', args.single])
            if args.json_output:
                cmd.append('--json_output')
            # Pass down the config file paths if they are provided
            if args.weights:
                cmd.extend(['--weights', args.weights])
            if args.subsets:
                cmd.extend(['--subsets', args.subsets])
            if args.doctor_id is not None:
                cmd.extend(['--doctor_id', str(args.doctor_id)])
            
            if not run_command(cmd, "Patient matching"):
                print("Pipeline failed at patient matching.")
                sys.exit(1)
        else:
            print("Skipping patient matching.")
        
        print("\nPipeline completed successfully!")
        print(f"Final output files are located in: {final_output_dir}")

    finally:
        # Clean up the temporary directory unless instructed otherwise
        if not args.keep_temp_dir:
            print(f"Cleaning up temporary directory: {work_dir}")
            shutil.rmtree(work_dir)
        else:
            print(f"Temporary directory kept for inspection: {work_dir}")

if __name__ == "__main__":
    # Import datetime for state saving
    from datetime import datetime
    main() 