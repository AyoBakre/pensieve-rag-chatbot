import os
import msal
import requests
import tempfile
from pathlib import Path

class SharePointExtractor:
    def __init__(self):
        # SharePoint credentials from environment variables
        self.site_url = os.environ.get("SHAREPOINT_SITE_URL")
        self.client_id = os.environ.get("SHAREPOINT_CLIENT_ID")
        self.client_secret = os.environ.get("SHAREPOINT_CLIENT_SECRET")
        self.tenant_id = os.environ.get("SHAREPOINT_TENANT_ID")
        self.folder_path = os.environ.get("SHAREPOINT_FOLDER_PATH", "")
        self.download_dir = 'downloaded_files'
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Graph API endpoint
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
        
        # Get access token for Microsoft Graph API
        self.access_token = self._get_access_token()
        
        # Get site ID for the SharePoint site
        self.site_id = self._get_site_id()
        
        # Get default document library ID
        self.drive_id = self._get_document_library_id()
        
        print(f"Initialized SharePoint extractor with folder path: '{self.folder_path}'")
        
    def _get_access_token(self):
        """Get an access token for Microsoft Graph API using MSAL"""
        try:
            # Configure MSAL application
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=authority
            )
            
            # The scope for Microsoft Graph API
            scopes = ["https://graph.microsoft.com/.default"]
            
            # Get token
            result = app.acquire_token_for_client(scopes=scopes)
            
            if "access_token" in result:
                print("✓ Successfully acquired access token for Microsoft Graph API")
                return result["access_token"]
            else:
                print(f"Error getting token: {result.get('error')}")
                print(f"Error description: {result.get('error_description')}")
                raise Exception("Failed to acquire access token")
                
        except Exception as e:
            print(f"Error in authentication: {str(e)}")
            raise
    
    def _get_site_id(self):
        """Get the SharePoint site ID using Microsoft Graph API"""
        if not self.access_token:
            raise Exception("No access token available")
            
        # Extract host name and site path from site URL
        host_name = self.site_url.split('//')[1].split('/')[0]
        site_path = '/'.join(self.site_url.split('//')[1].split('/')[1:])
        
        # API URL to get site ID
        site_id_url = f"{self.graph_endpoint}/sites/{host_name}:/{site_path}"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(site_id_url, headers=headers)
        
        if response.status_code == 200:
            site_id = response.json().get('id')
            print(f"✓ Successfully got site ID: {site_id}")
            return site_id
        else:
            print(f"Error getting site ID: {response.status_code}")
            print(response.text)
            raise Exception("Failed to get site ID")
    
    def _get_document_library_id(self):
        """Get the default document library (drive) ID using Microsoft Graph API"""
        if not self.access_token or not self.site_id:
            raise Exception("No access token or site ID available")
            
        # API URL to get drives in the site
        drives_url = f"{self.graph_endpoint}/sites/{self.site_id}/drives"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(drives_url, headers=headers)
        
        if response.status_code == 200:
            drives = response.json().get('value', [])
            if drives:
                # Get the first drive (usually the default document library)
                drive_id = drives[0].get('id')
                print(f"✓ Successfully got document library ID: {drive_id}")
                return drive_id
            else:
                raise Exception("No document libraries found in the site")
        else:
            print(f"Error getting document libraries: {response.status_code}")
            print(response.text)
            raise Exception("Failed to get document library ID")
    
    def list_files(self, folder_path=None):
        """List all files in the specified SharePoint folder using Microsoft Graph API"""
        if folder_path is None:
            folder_path = self.folder_path
            
        try:
            print(f"Listing files in folder: '{folder_path}'")
            
            # API URL to get items in the folder
            if not folder_path or folder_path == "":
                # Root folder
                items_url = f"{self.graph_endpoint}/drives/{self.drive_id}/root/children"
            else:
                # Specific folder
                items_url = f"{self.graph_endpoint}/drives/{self.drive_id}/root:/{folder_path}:/children"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            print(f"Making request to: {items_url}")
            response = requests.get(items_url, headers=headers)
            
            if response.status_code == 200:
                items = response.json().get('value', [])
                file_list = []
                
                for item in items:
                    if not item.get('folder'):  # Skip folders, only list files
                        file_info = {
                            "name": item.get('name'),
                            "url": item.get('webUrl'),
                            "size": item.get('size', 0),
                            "id": item.get('id')
                        }
                        file_list.append(file_info)
                        print(f"Found file: {file_info['name']}")
                
                print(f"Found {len(file_list)} files in folder '{folder_path}'")
                return file_list
            else:
                print(f"Error listing files (status {response.status_code}): {response.text}")
                # Try fallback to root folder
                if folder_path and folder_path != "":
                    print(f"Trying fallback to root folder...")
                    return self.list_files("")
                return []
                
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            return []
    
    def download_single_file(self, file_url, file_name):
        """Download a single file from SharePoint using Microsoft Graph API"""
        try:
            # Extract file ID from URL or use direct file ID if available
            if file_url.startswith("http"):
                # We need to search for the file first
                file_id = None
                # Try to find the file by name in the root or specified folder
                files = self.list_files(self.folder_path)
                for file in files:
                    if file["name"] == file_name:
                        file_id = file["id"]
                        break
                
                if not file_id:
                    print(f"Could not find file ID for {file_name}")
                    return None
            else:
                # Assume file_url is actually a file ID
                file_id = file_url
            
            # API URL to download the file
            download_url = f"{self.graph_endpoint}/drives/{self.drive_id}/items/{file_id}/content"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(download_url, headers=headers, stream=True)
            
            if response.status_code == 200:
                download_path = os.path.join(self.download_dir, file_name)
                
                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded: {file_name}")
                return download_path
            else:
                print(f"Error downloading {file_name}: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"Error downloading {file_name}: {str(e)}")
            return None
    
    def list_folders(self, folder_path=None):
        """List all folders in the specified SharePoint folder"""
        if folder_path is None:
            folder_path = self.folder_path
            
        try:
            # API URL to get items in the folder
            if folder_path == "":
                # Root folder
                items_url = f"{self.graph_endpoint}/drives/{self.drive_id}/root/children"
            else:
                # Specific folder
                items_url = f"{self.graph_endpoint}/drives/{self.drive_id}/root:/{folder_path}:/children"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(items_url, headers=headers)
            
            if response.status_code == 200:
                items = response.json().get('value', [])
                folder_list = []
                
                for item in items:
                    if item.get('folder'):  # Only list folders
                        folder_list.append({
                            "name": item.get('name'),
                            "path": f"{folder_path}/{item.get('name')}" if folder_path else item.get('name'),
                            "id": item.get('id')
                        })
                
                return folder_list
            else:
                print(f"Error listing folders: {response.status_code}")
                print(response.text)
                return []
                
        except Exception as e:
            print(f"Error listing folders: {str(e)}")
            return []
    
    def list_all_files_recursive(self, folder_path=None):
        """Recursively list all files from a SharePoint folder and its subfolders"""
        if folder_path is None:
            folder_path = self.folder_path
            
        all_files = []
        
        # List files in the current folder
        files = self.list_files(folder_path)
        all_files.extend(files)
        
        # List subfolders
        folders = self.list_folders(folder_path)
        
        # Process each subfolder recursively
        for folder in folders:
            subfolder_path = folder["path"]
            subfolder_files = self.list_all_files_recursive(subfolder_path)
            all_files.extend(subfolder_files)
            
        return all_files
    
    def download_files(self, folder_path=None):
        """Download all files from the specified SharePoint folder"""
        files = self.list_files(folder_path)
        downloaded_files = []
        
        for file_info in files:
            try:
                file_id = file_info["id"]
                file_name = file_info["name"]
                download_path = self.download_single_file(file_id, file_name)
                
                if download_path:
                    downloaded_files.append(download_path)
            except Exception as e:
                print(f"Error downloading {file_info.get('name')}: {str(e)}")
        
        return downloaded_files
    
    def download_folder_recursive(self, folder_path=None):
        """Recursively download all files from a SharePoint folder and its subfolders"""
        if folder_path is None:
            folder_path = self.folder_path
            
        all_downloaded_files = []
        
        # Download files in the current folder
        files = self.download_files(folder_path)
        all_downloaded_files.extend(files)
        
        # List subfolders
        folders = self.list_folders(folder_path)
        
        # Process each subfolder recursively
        for folder in folders:
            subfolder_path = folder["path"]
            subfolder_files = self.download_folder_recursive(subfolder_path)
            all_downloaded_files.extend(subfolder_files)
            
        return all_downloaded_files

if __name__ == "__main__":
    # For testing purposes - set these in your .env file or environment variables
    if not os.environ.get("SHAREPOINT_SITE_URL"):
        print("Setting environment variables for testing purposes only. In production, set these in your .env file.")
        os.environ["SHAREPOINT_SITE_URL"] = "your_sharepoint_site_url"
        os.environ["SHAREPOINT_CLIENT_ID"] = "your_client_id"
        os.environ["SHAREPOINT_CLIENT_SECRET"] = "your_client_secret"
        os.environ["SHAREPOINT_TENANT_ID"] = "your_tenant_id"
        os.environ["SHAREPOINT_FOLDER_PATH"] = ""  # Try listing from root
    
    extractor = SharePointExtractor()
    
    # List files in root
    root_files = extractor.list_files()
    print(f"Total files found in root: {len(root_files)}")
    
    # Try listing subfolders
    folders = extractor.list_folders()
    print(f"Folders in root:")
    for folder in folders:
        print(f" - {folder['name']}")
        
        # List files in this folder
        folder_files = extractor.list_files(folder['name'])
        print(f"   Files in {folder['name']}: {len(folder_files)}") 