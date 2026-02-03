import os
import time
from pathlib import Path
from typing import Dict, List

import boto3
import streamlit as st
from dotenv import load_dotenv

from utils.vecDb import LegalVectorDB
from graph_ver import run as graph_run


BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "extracted_text"
VECTOR_DB_DIR = BASE_DIR / "vectorDB"

TEXT_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)


def upload_to_s3(file_bytes: bytes, bucket: str, key: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=file_bytes)


def textract_pdf_from_s3(bucket: str, key: str, region: str) -> str:
    textract = boto3.client("textract", region_name=region)
    response = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = response["JobId"]

    while True:
        status = textract.get_document_text_detection(JobId=job_id)
        job_status = status["JobStatus"]
        if job_status in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(5)

    if job_status != "SUCCEEDED":
        raise RuntimeError(f"Textract job failed with status: {job_status}")

    blocks: List[Dict] = []
    next_token = None

    while True:
        if next_token:
            response = textract.get_document_text_detection(
                JobId=job_id, NextToken=next_token
            )
        else:
            response = textract.get_document_text_detection(JobId=job_id)

        blocks.extend(response["Blocks"])
        next_token = response.get("NextToken")
        if not next_token:
            break

    lines = [block["Text"] for block in blocks if block["BlockType"] == "LINE"]
    return "\n".join(lines)


def save_extracted_text(text: str, document_name: str) -> Path:
    path = TEXT_DIR / f"{document_name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def build_vector_db(text: str, document_name: str) -> Path:
    vecdb = LegalVectorDB()
    vecdb.ingest_document(text, source_name=document_name)
    vecdb.build_index()
    db_path = VECTOR_DB_DIR / document_name
    vecdb.save(str(db_path))
    return db_path


st.set_page_config(
    page_title="PPP Legal Analysis (LangGraph)",
    page_icon="📄",
    layout="wide"
)

st.title("PPP LLM Contract Analysis (LangGraph)")
st.caption("Upload a PDF, extract with Textract, build vector DB, then run LangGraph analysis.")

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

aws_region = os.getenv("AWS_REGION", "ap-south-1")
s3_bucket = os.getenv("S3_BUCKET", "")
s3_prefix = os.getenv("S3_PREFIX", "")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

document_name = st.text_input(
    "Document name",
    value=(
        uploaded_file.name.rsplit(".", 1)[0]
        if uploaded_file is not None
        else ""
    )
)

safe_name = document_name.strip()
existing_text_path = TEXT_DIR / f"{safe_name}.txt" if safe_name else None
existing_db_path = VECTOR_DB_DIR / safe_name if safe_name else None
ready_for_analysis = bool(
    safe_name
    and existing_text_path
    and existing_db_path
    and existing_text_path.exists()
    and existing_db_path.exists()
)

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

col1, col2 = st.columns(2)
with col1:
    run_extract = st.button("Extract + Build Vector DB")
with col2:
    if ready_for_analysis or st.session_state.analysis_ready:
        run_analysis = st.button("Run LangGraph Analysis")
    else:
        run_analysis = False

if run_extract:
    if uploaded_file is None:
        st.error("Please upload a PDF file.")
        st.stop()
    if not s3_bucket:
        st.error("Please provide the S3 bucket name for Textract.")
        st.stop()
    if not document_name.strip():
        st.error("Please provide a document name.")
        st.stop()

    s3_key_base = f"{safe_name}.pdf"
    if s3_prefix.strip():
        s3_key = f"{s3_prefix.strip().rstrip('/')}/{s3_key_base}"
    else:
        s3_key = s3_key_base

    with st.status("Uploading PDF to S3...", expanded=False) as status:
        file_bytes = uploaded_file.read()
        upload_to_s3(file_bytes, s3_bucket, s3_key, aws_region)
        status.update(label="PDF uploaded to S3.", state="complete")

    with st.status("Running AWS Textract...", expanded=False) as status:
        extracted_text = textract_pdf_from_s3(s3_bucket, s3_key, aws_region)
        status.update(label="Textract extraction completed.", state="complete")

    with st.status("Saving extracted text...", expanded=False) as status:
        text_path = save_extracted_text(extracted_text, safe_name)
        status.update(label=f"Saved text to {text_path}", state="complete")

    with st.status("Building vector database...", expanded=False) as status:
        db_path = build_vector_db(extracted_text, safe_name)
        status.update(label=f"Vector DB saved to {db_path}", state="complete")

    st.session_state.analysis_ready = True

    st.success("Extraction and vector DB build completed.")
    st.subheader("Stored Artifacts")
    st.write(f"Extracted text: `{text_path}`")
    st.write(f"Vector DB: `{db_path}`")

if run_analysis:
    if not document_name.strip():
        st.error("Please provide a document name to analyze.")
        st.stop()

    if not existing_text_path or not existing_text_path.exists():
        st.error(f"Extracted text not found: {existing_text_path}")
        st.stop()
    if not existing_db_path or not existing_db_path.exists():
        st.error(f"Vector DB not found: {existing_db_path}")
        st.stop()

    with st.status("Running LangGraph analysis...", expanded=False) as status:
        max_concurrency = int(os.getenv("OPENAI_MAX_CONCURRENCY", "2"))
        final_state = graph_run(safe_name, max_concurrency=max_concurrency)
        status.update(label="LangGraph analysis completed.", state="complete")

    st.success("Analysis completed.")
    st.subheader("Output Summary")
    st.write(final_state.summary)
