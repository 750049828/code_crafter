import os
import subprocess
import tempfile
import json
import re
from fastapi import FastAPI, UploadFile, File, Form
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
SUPPORTED_EXTENSIONS = {".py", ".sql"}

prompt_mapping = {
    "aws_to_databricks": AWS_TO_DATABRICKS_SYSTEM_INSTRUCTION,
    "databricks_to_aws": DATABRICKS_TO_AWS_SYSTEM_INSTRUCTION,
    "any_to_text_explanation": ANY_CODE_TO_TEXT_EXPLANATION_SYSTEM_INSTRUCTION,
    "sql_to_aws_or_databricks": SQL_TO_AWS_OR_DATABRICKS_SYSTEM_INSTRUCTION
}

def clean_gemini_response(response_text):
    """Removes markdown formatting and extracts valid JSON."""
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    return match.group(1) if match else response_text  # Extract JSON part if wrapped in triple backticks


def analyze_and_extract_code_from_repo(repo_url: str):
    """Clones a repo, scans for convertible files, and extracts code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        clone_path = os.path.join(temp_dir, "repo")
        subprocess.run(["git", "clone", repo_url, clone_path], check=True)

        total_files, convertible_files = 0, 0
        file_contents = {}

        for root, _, files in os.walk(clone_path):
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
        system_instruction = f"Convert {source_format} code to {target_format}."

    model = GenerativeModel(
        "gemini-1.5-pro-002",
        generation_config=generation_config,
        system_instruction=[system_instruction]
    )

    parts = [Part.from_text(code)]
    content = [Content(role="user", parts=parts)]

    response: GenerationResponse = model.generate_content(contents=content)
    cleaned_response = clean_gemini_response(response.text)

    return cleaned_response


@app.post("/convert")
async def convert(
    source_format: str = Form(...),
    target_format: str = Form(...),
    code: str = Form(None),
    repo_url: str = Form(None),
    file: UploadFile = File(None)
):
    response_data = {}

    if repo_url:
        total_files, convertible_files, file_contents = analyze_and_extract_code_from_repo(repo_url)
        response_data.update({"total_files": total_files, "convertible_files": convertible_files})
        
        converted_files = {}  # Store all converted files

        for file_name, content in file_contents.items():
            converted_code = convert_code(source_format, target_format, content)

            try:
                converted_code_json = json.loads(converted_code.strip("```json\n").strip("\n```"))  # Handle response parsing
            except json.JSONDecodeError:
                converted_code_json = {"error": f"Failed to parse response for {file_name}", "raw_response": converted_code}

            converted_files[file_name] = {
                "converted_code": converted_code_json.get("converted_code", ""),
                "confidence_score": converted_code_json.get("confidence_score", 0.0),
            }

        response_data["converted_files"] = converted_files  # Add all converted files
    elif file:
        file_name = file.filename
        original_code = await file.read()
        converted_code = convert_code(source_format, target_format, original_code.decode("utf-8"))

        # Parse the response
        try:
            converted_code_json = json.loads(clean_gemini_response(converted_code))
        except json.JSONDecodeError:
            return {"error": "Failed to parse response from Gemini API", "raw_response": converted_code}

        response_data.update({
            "file_name": file_name,
            "original_code": original_code.decode("utf-8"),
            "converted_code": converted_code_json.get("converted_code", ""),
            "confidence_score": converted_code_json.get("confidence_score", 0.0)
        })

    elif code:
        converted_code = convert_code(source_format, target_format, code)

        # Parse the response
        try:
            converted_code_json = json.loads(clean_gemini_response(converted_code))
        except json.JSONDecodeError:
            return {"error": "Failed to parse response from Gemini API", "raw_response": converted_code}

        response_data.update({
            "original_code": code,
            "converted_code": converted_code_json.get("converted_code", ""),
            "confidence_score": converted_code_json.get("confidence_score", 0.0)
        })

    else:
        return {"error": "No code, file, or repository provided"}

    return response_data
