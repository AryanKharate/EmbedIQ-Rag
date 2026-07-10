import requests
import uuid

API_BASE = "http://localhost:8000/api"

print("1. Listing documents...")
resp = requests.get(f"{API_BASE}/documents/")
print(resp.json())

print("\n2. Uploading a text file...")
files = {'file': ('test.txt', b"This is a test document with test content.", 'text/plain')}
resp = requests.post(f"{API_BASE}/documents/upload", files=files)
print(resp.json())
doc_id = resp.json().get('id')

print("\n3. Listing documents again...")
resp = requests.get(f"{API_BASE}/documents/")
print(resp.json())

print("\n4. Toggling active status to false...")
resp = requests.patch(f"{API_BASE}/documents/{doc_id}/toggle", json={"is_active": False})
print(resp.json())

print("\n5. Deleting document...")
resp = requests.delete(f"{API_BASE}/documents/{doc_id}")
print(resp.json())

print("\n6. Listing documents final...")
resp = requests.get(f"{API_BASE}/documents/")
print(resp.json())
