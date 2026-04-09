import argparse
import json
import random
import sys

def distribute_ids_to_doctors(file_path: str, num_doctors: int, refresh: bool = False, doctor_id_range: tuple = (1000, 9999)) -> str:
    """Distributes patient IDs from a text file among a specified number of doctors.

    Args:
        file_path: The path to the text file containing patient IDs, one per line.
        num_doctors: The number of doctors to distribute the IDs among.
        refresh: A boolean indicating whether to include a 'refresh' flag in the output JSON.
        doctor_id_range: A tuple specifying the minimum and maximum doctor ID to generate.

    Returns:
        A JSON string representing the distribution of patient IDs to doctors.
    """
    if num_doctors <= 0:
        print("Error: num_doctors must be a positive integer.", file=sys.stderr)
        return json.dumps({})

    patient_ids = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    patient_ids.append(stripped_line)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
        return json.dumps({})
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        return json.dumps({})

    if not patient_ids and num_doctors > 0:
        print("Warning: No patient IDs found in the file. Generating requests with empty patient lists.", file=sys.stderr)

    if (doctor_id_range[1] - doctor_id_range[0] + 1) < num_doctors:
        print(f"Warning: Doctor ID range ({doctor_id_range}) is too small to generate {num_doctors} unique IDs. Adjusting range or reducing num_doctors might be needed.", file=sys.stderr)
        generated_doctor_ids = random.sample(range(doctor_id_range[0], doctor_id_range[1] + 1), min(num_doctors, (doctor_id_range[1] - doctor_id_range[0] + 1)))
    else:
        generated_doctor_ids = random.sample(range(doctor_id_range[0], doctor_id_range[1] + 1), num_doctors)


    doctor_assignments = {doc_id: [] for doc_id in generated_doctor_ids}

    if patient_ids:
        for i, patient_id in enumerate(patient_ids):
            doctor_id_index = i % num_doctors
            assigned_doctor_id = generated_doctor_ids[doctor_id_index]
            doctor_assignments[assigned_doctor_id].append(patient_id)

    requests_array = []
    for doc_id in generated_doctor_ids:
        requests_array.append({
            "doctor_id": doc_id,
            "patient_ids": doctor_assignments[doc_id]
        })

    json_body = {
        "requests": requests_array,
        "refresh": refresh
    }

    return json.dumps(json_body, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribute patient IDs from a text file among doctors and generate a JSON output.")
    parser.add_argument("input_file", help="Path to the input text file containing patient IDs (one per line).")
    parser.add_argument("num_doctors", type=int, help="The number of doctors to distribute IDs among.")
    parser.add_argument("-o", "--output_file", default="output.json", help="Path to the output JSON file (default: output.json).")
    parser.add_argument("--refresh", action="store_true", help="Include a 'refresh' flag in the output JSON.")
    parser.add_argument("--doctor_id_range", type=int, nargs=2, default=(1000, 9999), metavar=('MIN', 'MAX'),
                        help="Range for generating doctor IDs (inclusive, default: 1000 9999).")

    args = parser.parse_args()

    json_output = distribute_ids_to_doctors(
        args.input_file,
        args.num_doctors,
        args.refresh,
        tuple(args.doctor_id_range)
    )

    try:
        with open(args.output_file, "w") as f:
            f.write(json_output)
        print(f"Output saved to {args.output_file}")
    except IOError as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)