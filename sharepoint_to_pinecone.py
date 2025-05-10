import os
import json
import time
import uuid
from openai import OpenAI  # Updated import for OpenAI API
import datetime
import gc  # For garbage collection
from pathlib import Path
from sharepoint_extractor import SharePointExtractor
from document_processor import DocumentProcessor
from pinecone import Pinecone  # Updated Pinecone import

# Enable aggressive garbage collection for better memory management
gc.enable()

# Global variables for tracking progress
total_files_processed = 0
total_chunks_uploaded = 0
files_in_current_batch = 0
sharepoint_site_url = ""  # Will be set in main()

def process_folder(folder_path, extractor, processor, pinecone_success, pinecone_index, openai_client, embedding_model, supported_extensions, excluded_extensions, max_files_per_batch, progress_file, processed_folders, processed_files):
    """Process a folder and all its subfolders recursively"""
    global total_files_processed, total_chunks_uploaded, files_in_current_batch
    
    try:
        print(f"\n[DEBUG] Starting to process folder: {folder_path or 'ROOT'}")
        
        # Skip if already processed
        if folder_path in processed_folders:
            print(f"Skipping already processed folder: {folder_path or 'ROOT'}")
            return
            
        print(f"\n> Processing folder: {folder_path or 'ROOT'}")
        
        # Step 1: Get files in this folder
        print(f"Getting files from: {folder_path or 'ROOT'}")
        folder_files = extractor.list_files(folder_path)
        print(f"Found {len(folder_files)} files")
        
        # Step 2: Filter supported files
        supported_files = []
        for file_info in folder_files:
            try:
                file_name = file_info["name"]
                file_id = file_info["id"]
                
                # Generate a unique file identifier
                file_identifier = f"{folder_path}|{file_id}"
                
                # Skip if already processed
                if file_identifier in processed_files:
                    print(f"- Skipping {file_name} (already processed)")
                    continue
                
                file_extension = os.path.splitext(file_name)[1].lower()
                
                if file_extension in excluded_extensions:
                    print(f"- Skipping {file_name} (excluded type)")
                    continue
                    
                if file_extension in supported_extensions:
                    supported_files.append({
                        **file_info, 
                        "identifier": file_identifier,
                        "folder_path": folder_path
                    })
                    print(f"+ Adding {file_name} for processing")
            except Exception as e:
                print(f"[DEBUG] Error filtering file: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Found {len(supported_files)} supported documents to process")
        
        # Step 3: Process each file one by one
        for file_idx, file_info in enumerate(supported_files):
            try:
                # Check if we've reached the maximum files per batch
                if files_in_current_batch >= max_files_per_batch:
                    print(f"\n=== Reached maximum of {max_files_per_batch} files per batch ===")
                    print("Clearing memory and resetting batch counter...")
                    # Force aggressive garbage collection
                    gc.collect()
                    # Reset batch counter
                    files_in_current_batch = 0
                    
                    # Save progress
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'processed_folders': list(processed_folders),
                            'processed_files': list(processed_files)
                        }, f)
                    print("Saved progress and cleared memory, continuing with next batch")
                
                file_id = file_info["id"]
                file_name = file_info["name"]
                file_identifier = file_info["identifier"]
                folder_path = file_info["folder_path"]
                file_size = file_info["size"]
                file_size_mb = file_size / (1024 * 1024)
                file_extension = os.path.splitext(file_name)[1].lower()
                
                print(f"\n[{file_idx+1}/{len(supported_files)}] Processing: {file_name} ({file_size_mb:.2f} MB)")
                print(f"  From folder: {folder_path or 'ROOT'}")
                
                # Save progress before processing large files to make sure we don't reprocess
                if file_size_mb > 1.0:  # Save progress for files > 1MB
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'processed_folders': list(processed_folders),
                            'processed_files': list(processed_files)
                        }, f)
                    print("  Saved progress before processing large file")
                
                # Phase 1: Download file
                print("  Downloading file...")
                download_path = extractor.download_single_file(file_id, file_name)
                if not download_path:
                    print("  Failed to download, skipping.")
                    continue
                
                # Phase 2: Process into chunks
                print("  Chunking document...")
                chunks = []
                try:
                    if file_extension in ['.docx', '.doc']:
                        chunks = process_word_document(download_path, file_name)
                    elif file_extension == '.pdf':
                        # Use a smaller chunk size for PDFs to conserve memory
                        chunks = processor.process_single_document(download_path, chunk_size=500)
                        # Force garbage collection after PDF processing
                        gc.collect()
                    else:
                        chunks = processor.process_single_document(download_path)
                except Exception as e:
                    print(f"  Error processing document: {str(e)}")
                    
                if not chunks:
                    print("  No text extracted, skipping.")
                    os.remove(download_path)
                    # Mark as processed anyway to avoid reprocessing errors
                    processed_files.add(file_identifier)
                    continue
                
                print(f"  Created {len(chunks)} text chunks")
                
                # Phase 3: Generate embeddings and upload to Pinecone
                if pinecone_success:
                    print("  Generating embeddings and uploading to Pinecone...")
                    chunks_uploaded = upload_chunks_to_pinecone(
                        chunks, 
                        file_info, 
                        folder_path, 
                        sharepoint_site_url, 
                        openai_client, 
                        embedding_model,
                        pinecone_index
                    )
                    total_chunks_uploaded += chunks_uploaded
                    print(f"  Uploaded {chunks_uploaded} vectors to Pinecone")
                else:
                    print("  Skipping Pinecone upload (offline mode)")
                
                # Clean up downloaded file
                os.remove(download_path)
                print("  Cleaned up downloaded file")
                
                # Update stats
                total_files_processed += 1
                files_in_current_batch += 1
                processed_files.add(file_identifier)
                print(f"  Completed processing {file_name}")
                
                # Force garbage collection after each file
                print("  Clearing memory...")
                del chunks
                gc.collect()
                
                # Save progress after each file for large files
                if file_size_mb > 1.0:
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'processed_folders': list(processed_folders),
                            'processed_files': list(processed_files)
                        }, f)
                    print("  Saved progress after processing large file")
                
            except Exception as e:
                print(f"  Error processing file {file_name}: {str(e)}")
                # Mark as processed anyway to avoid reprocessing errors
                processed_files.add(file_identifier)
                # Save progress after error to avoid reprocessing
                with open(progress_file, 'w') as f:
                    json.dump({
                        'processed_folders': list(processed_folders),
                        'processed_files': list(processed_files)
                    }, f)
                # Continue with next file even if one fails
                import traceback
                traceback.print_exc()
                continue
        
        # Get subfolders and process them recursively
        print(f"[DEBUG] Getting subfolders for: {folder_path or 'ROOT'}")
        subfolders = extractor.list_folders(folder_path)
        print(f"Found {len(subfolders)} subfolders in {folder_path or 'ROOT'}")
        
        if len(subfolders) == 0:
            # This is a leaf folder, mark it as processed
            processed_folders.add(folder_path)
            print(f"Marking leaf folder as processed: {folder_path or 'ROOT'}")
            
            # Save progress after processing leaf folder
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_folders': list(processed_folders),
                    'processed_files': list(processed_files)
                }, f)
        else:
            # Process each subfolder recursively
            for subfolder in subfolders:
                try:
                    subfolder_path = subfolder["path"]
                    print(f"[DEBUG] Processing subfolder: {subfolder_path}")
                    process_folder(
                        subfolder_path, 
                        extractor, 
                        processor, 
                        pinecone_success, 
                        pinecone_index, 
                        openai_client, 
                        embedding_model, 
                        supported_extensions, 
                        excluded_extensions, 
                        max_files_per_batch, 
                        progress_file, 
                        processed_folders, 
                        processed_files
                    )
                except Exception as e:
                    print(f"[DEBUG] Error processing subfolder {subfolder.get('name', 'unknown')}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
            # After all subfolders are processed, mark this folder as processed
            processed_folders.add(folder_path)
            print(f"Marking folder with subfolders as processed: {folder_path or 'ROOT'}")
            
            # Save progress after processing all subfolders
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_folders': list(processed_folders),
                    'processed_files': list(processed_files)
                }, f)
    except Exception as e:
        print(f"[DEBUG] Process folder error for {folder_path or 'ROOT'}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting SharePoint to Pinecone Migration...")
    
    # Access global variables
    global total_files_processed, total_chunks_uploaded, files_in_current_batch, sharepoint_site_url
    
    # Hard-coded values
    # SharePoint credentials
    sharepoint_site_url = os.environ.get("SHAREPOINT_SITE_URL", "your_sharepoint_site_url")
    sharepoint_client_id = os.environ.get("SHAREPOINT_CLIENT_ID", "your_client_id")
    sharepoint_client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET", "your_client_secret")
    sharepoint_tenant_id = os.environ.get("SHAREPOINT_TENANT_ID", "your_tenant_id")
    sharepoint_folder_path = os.environ.get("SHAREPOINT_FOLDER_PATH", "")  # Empty string for root access
    
    # Pinecone configuration
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "your_pinecone_api_key")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "sixpoint-rag")
    
    # OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
    
    # File processing config
    max_file_size_mb = 30  # Reduced from 50 to 30 MB to help with memory issues
    supported_extensions = ['.pdf', '.docx', '.doc', '.pptx']  # Document types to process
    excluded_extensions = ['.xlsx', '.xls', '.csv', '.url']  # File types to skip
    
    # Memory management settings
    max_files_per_batch = 3  # Reduced from 5 to 3 files per batch
    
    # Environment setup
    os.environ["SHAREPOINT_SITE_URL"] = sharepoint_site_url
    os.environ["SHAREPOINT_CLIENT_ID"] = sharepoint_client_id
    os.environ["SHAREPOINT_CLIENT_SECRET"] = sharepoint_client_secret
    os.environ["SHAREPOINT_TENANT_ID"] = sharepoint_tenant_id
    os.environ["SHAREPOINT_FOLDER_PATH"] = sharepoint_folder_path
    
    # Create status/progress file to enable resume functionality
    progress_file = "migration_progress.json"
    processed_folders = set()
    processed_files = set()
    
    # Create progress file if it doesn't exist
    if not os.path.exists(progress_file):
        with open(progress_file, 'w') as f:
            json.dump({
                'processed_folders': [],
                'processed_files': []
            }, f)
    
    # Load progress if exists
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                # Only keep the processed files, not folders
                processed_folders = set(progress_data.get('processed_folders', []))
                processed_files = set(progress_data.get('processed_files', []))
            print(f"Loaded progress: {len(processed_folders)} folders and {len(processed_files)} files already processed")
        except Exception as e:
            print(f"Error loading progress file: {str(e)}")
            # Create a blank progress file
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_folders': [],
                    'processed_files': []
                }, f)
            processed_files = set()
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)
    embedding_model = "text-embedding-3-small"
    
    # Create directory for local storage
    os.makedirs('downloaded_files', exist_ok=True)
    
    try:
        # Initialize SharePoint extractor
        print("\n=== Initializing SharePoint connection ===")
        extractor = SharePointExtractor()
        
        # Initialize document processor
        processor = DocumentProcessor(max_file_size_mb=max_file_size_mb)
        print(f"Document processor configured:")
        print(f"- Max file size: {max_file_size_mb} MB")
        print(f"- Supported file types: {', '.join(supported_extensions)}")
        print(f"- Max files per batch: {max_files_per_batch}")
        
        # Initialize Pinecone
        print("\n=== Initializing Pinecone ===")
        pinecone_success, pinecone_index = initialize_pinecone(pinecone_api_key, pinecone_index_name)
        
        # Process files in root and all subfolders recursively
        print("\n=== Starting file processing pipeline ===")
        
        # Start with the root folder and process recursively
        process_folder(
            "", 
            extractor, 
            processor, 
            pinecone_success, 
            pinecone_index, 
            openai_client, 
            embedding_model, 
            supported_extensions, 
            excluded_extensions, 
            max_files_per_batch, 
            progress_file, 
            processed_folders, 
            processed_files
        )
        
        # Print final stats
        print("\n=== Processing Complete ===")
        print(f"Total files processed: {total_files_processed}")
        print(f"Total chunks uploaded to Pinecone: {total_chunks_uploaded}")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        import traceback
        traceback.print_exc()

def initialize_pinecone(api_key, index_name):
    """Initialize Pinecone and connect to index"""
    try:
        print(f"Connecting to Pinecone index: {index_name}")
        
        # Initialize Pinecone with the latest client
        print(f"  Creating Pinecone client with API key: {api_key[:5]}...")
        pc = Pinecone(api_key=api_key)
        
        # Get the index
        print(f"  Connecting to index: {index_name}")
        index = pc.Index(index_name)
        
        # Test connection with simple stats call
        print(f"  Testing connection with describe_index_stats call...")
        stats = index.describe_index_stats()
        print(f"Successfully connected to Pinecone - Vector count: {stats.get('total_vector_count', 0)}")
        return True, index
    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")
        import traceback
        print("Detailed error information:")
        traceback.print_exc()
        print("Will continue in offline mode (storing chunks locally only)")
        return False, None

def process_word_document(file_path, file_name):
    """Process Word documents with memory efficient approach"""
    try:
        import docx
        print("  Opening Word document...")
        doc = docx.Document(file_path)
        
        chunk_size = 5  # Number of paragraphs per chunk
        chunks = []
        paragraphs = []
        
        print("  Extracting paragraphs...")
        # Get non-empty paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        # Release document object to free memory
        del doc
        gc.collect()
        
        # Create chunks from paragraphs
        total_chunks = (len(paragraphs) + chunk_size - 1) // chunk_size
        print(f"  Found {len(paragraphs)} paragraphs, creating {total_chunks} chunks...")
        
        for i in range(0, len(paragraphs), chunk_size):
            chunk_paragraphs = paragraphs[i:i+chunk_size]
            combined_text = "\n\n".join(chunk_paragraphs)
            if combined_text.strip():
                chunk_id = i // chunk_size
                chunks.append({
                    "text": combined_text,
                    "metadata": {
                        "source": file_name,
                        "chunk_id": chunk_id,
                        "total_chunks": total_chunks
                    }
                })
                
                # Force garbage collection after creating each chunk
                if chunk_id % 10 == 0:  # Every 10 chunks
                    gc.collect()
        
        # Free memory
        del paragraphs
        gc.collect()
        
        return chunks
    
    except Exception as e:
        print(f"Error processing Word document: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def upload_chunks_to_pinecone(chunks, file_info, folder_path, sharepoint_site_url, openai_client, embedding_model, pinecone_index):
    """Upload document chunks to Pinecone with memory-efficient batching"""
    try:
        # Process in very small batches to minimize memory usage
        batch_size = 1  # Reduced from 2 to 1 to minimize memory usage
        uploaded_count = 0
        
        # Add document metadata
        file_name = file_info["name"]
        file_extension = os.path.splitext(file_name)[1].lower()
        file_size = file_info.get("size", 0)
        file_size_mb = file_size / (1024 * 1024) if file_size else 0
        processing_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        namespace = "sharepoint-documents"
        
        # Estimate total pages based on total chunks
        total_chunks = len(chunks)
        estimated_pages = max(1, total_chunks // 2)
        
        print(f"  Processing {len(chunks)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i+batch_size, len(chunks))
            batch = chunks[i:i+batch_size]
            
            print(f"  Batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            try:
                # Generate embeddings
                texts = [chunk["text"] for chunk in batch]
                response = openai_client.embeddings.create(
                    input=texts,
                    model=embedding_model
                )
                embeddings = [item.embedding for item in response.data]
                
                # Prepare vectors
                records = []
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunk_id = chunk['metadata']['chunk_id']
                    total_chunks = chunk['metadata']['total_chunks']
                    
                    # Calculate position in document
                    position_ratio = chunk_id / total_chunks if total_chunks > 0 else 0
                    if position_ratio < 0.33:
                        position_in_doc = "beginning"
                    elif position_ratio < 0.66:
                        position_in_doc = "middle"
                    else:
                        position_in_doc = "end"
                    
                    # Estimate page number
                    estimated_page = int(position_ratio * estimated_pages) + 1
                    
                    # Create vector with rich metadata
                    record = {
                        "id": f"{file_name}_{chunk_id}_{uuid.uuid4()}",
                        "values": embedding,
                        "metadata": {
                            # Content
                            "text": chunk["text"],
                            "source": chunk["metadata"]["source"],
                            "chunk_id": chunk_id,
                            "total_chunks": total_chunks,
                            
                            # File metadata
                            "file_extension": file_extension,
                            "file_type": file_extension.replace('.', ''),
                            "file_size_bytes": file_size,
                            "file_size_mb": round(file_size_mb, 2),
                            "sharepoint_url": file_info.get("url", ""),
                            
                            # Location metadata
                            "folder_path": folder_path,
                            "sharepoint_site": sharepoint_site_url,
                            
                            # Processing metadata
                            "processing_date": processing_date,
                            "estimated_page": estimated_page,
                            "estimated_total_pages": estimated_pages,
                            "position_in_document": position_in_doc,
                            "content_length": len(chunk["text"])
                        }
                    }
                    records.append(record)
                
                # Upsert to Pinecone using the latest client syntax
                pinecone_index.upsert(
                    vectors=records,
                    namespace=namespace
                )
                uploaded_count += len(records)
                
                # Clean up to minimize memory usage
                del texts
                del records
                del embeddings
                del response
                gc.collect()
                
                # Avoid rate limiting and add brief pause for memory to stabilize
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error processing batch: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return uploaded_count
        
    except Exception as e:
        print(f"Error uploading to Pinecone: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    main() 