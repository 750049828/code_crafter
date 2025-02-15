import os
import io
import subprocess
import tempfile
import json
import re
import zipfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content, GenerationResponse
from prompts import (
    ANY_CODE_TO_TEXT_EXPLANATION_SYSTEM_INSTRUCTION,
    DATABRICKS_TO_AWS_SYSTEM_INSTRUCTION,
    AWS_TO_DATABRICKS_SYSTEM_INSTRUCTION,
    SQL_TO_AWS_OR_DATABRICKS_SYSTEM_INSTRUCTION
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = {".py", ".sql"}

prompt_mapping = {
    "aws_to_databricks": AWS_TO_DATABRICKS_SYSTEM_INSTRUCTION,
    "databricks_to_aws": DATABRICKS_TO_AWS_SYSTEM_INSTRUCTION,
    "any_to_text_explanation": ANY_CODE_TO_TEXT_EXPLANATION_SYSTEM_INSTRUCTION,
    "sql_to_aws_or_databricks": SQL_TO_AWS_OR_DATABRICKS_SYSTEM_INSTRUCTION
}

def clean_gemini_response(response_text):
    """Extract JSON from Gemini response, ensuring valid JSON output."""
    
    # If already a dict, return as is
    if isinstance(response_text, dict):
        return response_text  # ✅ Prevents TypeError
    
    # If response is a string, try extracting JSON from markdown format
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    extracted_json = match.group(1) if match else response_text.strip()

    # Validate JSON before returning
    try:
        return json.loads(extracted_json)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "raw_response": extracted_json}


def clone_repo(repo_url: str, temp_dir: str):
    """Clones a repo into a temporary directory."""
    try:
        repo_path = os.path.join(temp_dir, "repo")
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    except Exception as e:
        print(f"Exception Occured. Could not clone repo: {e}")
    print(f"Cloned Repo at path: {repo_path}")
    return repo_path


def analyze_and_extract_code_from_repo(repo_path: str):
    """Scans for convertible files and extracts code."""
    total_files, convertible_files = 0, 0
    file_contents = {}

    for root, _, files in os.walk(repo_path):
        for file in files:
            total_files += 1
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                convertible_files += 1
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    file_contents[file] = f.read()

    return total_files, convertible_files, file_contents


def convert_code(source_format: str, target_format: str, code: str):
    vertexai.init()
    generation_config = {"max_output_tokens": 8192, "temperature": 0.3, "top_p": 0.95}

    # Determine the correct prompt
    if "text" in target_format.lower():
        system_instruction = prompt_mapping["any_to_text_explanation"]
    elif "aws" in source_format.lower() and "databricks" in target_format.lower():
        system_instruction = prompt_mapping["aws_to_databricks"]
    elif "databricks" in source_format.lower() and "aws" in target_format.lower():
        system_instruction = prompt_mapping["databricks_to_aws"].format(
            sourceCodeFormat=source_format, targetCodeFormat=target_format
        )
    elif "sql" in source_format.lower():
        system_instruction = prompt_mapping["sql_to_aws_or_databricks"].format(
            sourceCodeFormat=source_format, targetCodeFormat=target_format
        )
    else:
        system_instruction = f"""Convert {source_format} code to {target_format}. **Return the output as json {{'converted_code':'','confidence_score': ''}}**"""

    model = GenerativeModel(
        "gemini-1.5-pro-002",
        generation_config=generation_config,
        system_instruction=[system_instruction]
    )

    parts = [Part.from_text(code)]
    content = [Content(role="user", parts=parts)]

    response: GenerationResponse = model.generate_content(contents=content)
    cleaned_response = clean_gemini_response(response.text)
    #print(response.text)
    return cleaned_response


def save_converted_code_to_repo(target_repo_path, converted_files):
    """Saves converted code into the target repo, commits, and pushes changes."""
    for file_name, file_data in converted_files.items():
        converted_code = file_data["converted_code"]
        file_path = os.path.join(target_repo_path, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(converted_code)

    # Git commit and push
    subprocess.run(["git", "add", "."], cwd=target_repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Added converted code"], cwd=target_repo_path, check=True)
    subprocess.run(["git", "push"], cwd=target_repo_path, check=True)

#added function to convert list of converted code into zip file. not yet integrated with rest of the code
def save_converted_code_to_zip(converted_files):
    """Saves converted repo as a zip of files and returns the zip file"""
     # Create an in-memory bytes buffer
    zip_buffer = io.BytesIO()

    # Create a zip archive in memory
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_name, file_data in converted_files.items():
            zipf.writestr(file_name, file_data["converted_code"])

    # Move the cursor to the beginning of the buffer
    zip_buffer.seek(0)

    return zip_buffer


@app.post("/convert")
async def convert(
    repoPath1: str = Form(""),
    repoPath2: str = Form(""),
    repositoryType1: str =  Form(""),
    repositoryType2: str = Form(""),
    source_format: str = Form(...),
    target_format: str = Form(...),
    source_code: str = Form(None), #changed variable name to source_code from code
    file: UploadFile = File(None)
):
    response_data = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        source_repo_path = clone_repo(repoPath1, temp_dir) if repoPath1 else None
        target_repo_path = clone_repo(repoPath2, temp_dir) if repoPath2 else None

        converted_files = {}

        if source_repo_path:
            total_files, convertible_files, file_contents = analyze_and_extract_code_from_repo(source_repo_path)
            response_data.update({"total_files": total_files, "convertible_files": convertible_files})

            for file_name, content in file_contents.items():
                print(f"Converting {file_name} from {source_format} to {target_format}")
                converted_code = convert_code(source_format, target_format, content)
                try:
                    converted_code_json = clean_gemini_response(converted_code)
                except json.JSONDecodeError:
                    converted_code_json = {"error": f"Failed to parse response for {file_name}", "raw_response": converted_code}

                converted_files[file_name] = {
                    "converted_code": converted_code_json.get("converted_code", ""),
                    "confidence_score": converted_code_json.get("confidence_score", 0.0),
                }
            response_data["converted_files"] = converted_files 

        elif file:
            file_name = file.filename
            original_code = await file.read()
            print(f"Converting {file_name} from {source_format} to {target_format}")
            converted_code = convert_code(source_format, target_format, original_code.decode("utf-8"))

            try:
                converted_code_json = clean_gemini_response(converted_code)
            except json.JSONDecodeError:
                return {"error": "Failed to parse response from Gemini API", "raw_response": converted_code}

            response_data.update({
            "file_name": file_name,
            "original_code": original_code.decode("utf-8"),
            "converted_code": converted_code_json.get("converted_code", ""),
            "confidence_score": converted_code_json.get("confidence_score", 0.0)
             })

        elif source_code:
            print(f"Converting pasted code from {source_format} to {target_format}")
            converted_code = convert_code(source_format, target_format, source_code)
            try:
                converted_code_json = clean_gemini_response(converted_code)
            except json.JSONDecodeError:
                return {"error": "Failed to parse response from Gemini API", "raw_response": converted_code}

            response_data.update({
            "original_code": source_code,
            "converted_code": converted_code_json.get("converted_code", ""),
            "confidence_score": converted_code_json.get("confidence_score", 0.0)
            })
        
        else:
            return {"error": "No code, file, or repository provided"}

        if target_repo_path:
            save_converted_code_to_repo(target_repo_path, converted_files)
            response_data["message"] = f"Converted code saved to {repoPath2}"
    print("Sending response back!")
    return  response_data
