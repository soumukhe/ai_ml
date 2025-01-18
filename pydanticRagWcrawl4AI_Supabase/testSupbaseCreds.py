from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# Debug environment variables
print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_SERVICE_KEY:", os.getenv("SUPABASE_SERVICE_KEY"))



# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Debug table access
try:
    response = supabase.table("site_pages").select("*").execute()
    print("Supabase connection successful. Response:")
    print(response)
except Exception as e:
    print(f"Error accessing Supabase: {e}")