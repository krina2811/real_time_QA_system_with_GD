import base64
import io
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from datetime import datetime, timezone
import os
import json


class MyGDrive:
    def __init__(self, service_account_json_key):
        # ... existing initialization code ...
        scope = ['https://www.googleapis.com/auth/drive']
        # service_account_json_key = 'service-account-key.json'
        credentials = service_account.Credentials.from_service_account_file(
                                      filename=service_account_json_key,
                                      scopes=scope)
        self.service = build('drive', 'v3', credentials=credentials)
        self.timestamp_file = "last_processed_time.json"
        self.state = self.load_state()
        self.last_processed_time = self.state.get('last_processed_time', self.get_default_timestamp())
        print("last processed time.........................", self.last_processed_time)
        self.processed_files_file = "processed_files_list.json"
        self.processed_files = self.load_processed_files()
        print("my drive object is created..............")
    
    def get_default_timestamp(self):
        """Get a default timestamp (current time in ISO format)."""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


    def load_state(self):
        """Load the state (including first_run and last_processed_time) from file."""
        try:
            if os.path.exists(self.timestamp_file):
                with open(self.timestamp_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
        return {'first_run': True, 'last_processed_time': self.get_default_timestamp()}

    def save_state(self, last_processed_time=None):
        """Save the current state to file."""
        try:
            current_state = self.state.copy()
            if last_processed_time:
                current_state['last_processed_time'] = last_processed_time
            current_state['first_run'] = False  # Always set first_run to False after saving
            

            print("in save process time.....................................................", current_state)
            with open(self.timestamp_file, 'w') as f:
                json.dump(current_state, f)
            self.state = current_state
        except Exception as e:
            print(f"Error saving state: {e}")


    def load_processed_files(self):
        """Load the set of processed file IDs from disk."""
        try:
            if os.path.exists(self.processed_files_file):
                with open(self.processed_files_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            print(f"Error loading processed files: {e}")
            return set()

    def save_processed_files(self):
        """Save the current set of processed file IDs to disk."""
        try:
            with open(self.processed_files_file, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            print(f"Error saving processed files: {e}")

    def is_file_processed(self, file_id):
        """Check if a file has been processed."""
        return file_id in self.processed_files

    def mark_file_as_processed(self, file_id):
        """Mark a file as processed and save the updated state."""

        print("------------------- make file ______________________")
        try:
            # Get the file's metadata to update last processed time
            file = self.service.files().get(
                fileId=file_id,
                fields="modifiedTime"
            ).execute()
            
            # Add file_id to processed files set
            self.processed_files.add(file_id)
            self.save_processed_files()
            
            # Update last processed time if this file is more recent
            if file.get('modifiedTime', '') > self.last_processed_time:
                self.last_processed_time = file['modifiedTime']
            # self.save_state(self.last_processed_time)
        except Exception as e:
            print(f"Error marking file as processed: {e}")

    def get_files(self):
        """
        Fetches all files on first run, then only new/modified files afterward.
        Now includes checking for already processed files.
        """
        
        print("In get files last processed time")
        try:
            if self.state.get('first_run', True):
                current_last_processed = self.last_processed_time
                print(f"🚀 First run: Fetching all files from Google Drive: {current_last_processed}")
                query = ""
            else:
                self.state = self.load_state()
                current_last_processed = self.state.get('last_processed_time')
                self.last_processed_time = self.get_default_timestamp()
                print(f"🔄 Checking for new/modified files since: {current_last_processed}")
                query = (
                    "("
                    f"createdTime > '{current_last_processed}' or "
                    f"modifiedTime > '{current_last_processed}'"
                    ")"
                )
            self.save_state(self.last_processed_time)
            result = self.service.files().list(
                q=query,
                pageSize=100,
                fields="files(id, name, createdTime, modifiedTime)",
                orderBy="createdTime desc"
            ).execute()

            files = result.get('files', [])
            
            # Filter out already processed files
            unprocessed_files = [
                {
                    'id': file.get('id'),
                    'name': file.get('name'),
                    'createdTime': file.get('createdTime'),
                    'modifiedTime': file.get('modifiedTime')
                }
                for file in files 
                if not self.is_file_processed(file.get('id'))
            ]

            if not unprocessed_files:
                print("✅ No new unprocessed files detected.")
                return []

            return unprocessed_files

        except Exception as e:
            print(f"❌ Error fetching files: {str(e)}")
            return []


    def download_pdf_to_memory(self, fileId: str):
        try:
            request = self.service.files().get_media(fileId=fileId)
            file_stream = io.BytesIO()  # Create a memory buffer to hold the file
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            file_stream.seek(0)  # Reset the stream position to the beginning
            return file_stream
        except HttpError as error:
            print(F'An error occurred: {error}')
