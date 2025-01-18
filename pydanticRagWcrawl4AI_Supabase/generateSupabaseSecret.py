import jwt
import datetime
import secrets

# Generate a secure JWT_SECRET
def generate_jwt_secret():
    return secrets.token_urlsafe(64)  # Generate a random, URL-safe 64-character string

# Generate a JWT token with a specific role (e.g., "service_role" or "anon")
def generate_key(secret, role):
    payload = {
        "role": role,  # Role can be "service_role" or "anon"
        "iss": "supabase",  # Issuer must be "supabase"
        "iat": int(datetime.datetime.now().timestamp()),  # Issued at
        "exp": int((datetime.datetime.now() + datetime.timedelta(days=365)).timestamp())  # Expiry
    }
    return jwt.encode(payload, secret, algorithm="HS256")

if __name__ == "__main__":
    # Generate the JWT_SECRET
    jwt_secret = generate_jwt_secret()

    # Generate SERVICE_ROLE_KEY and ANON_KEY
    service_role_key = generate_key(jwt_secret, "service_role")
    anon_key = generate_key(jwt_secret, "anon")

    # Print the generated secrets
    print(f"JWT_SECRET: {jwt_secret}")
    print(f"SERVICE_ROLE_KEY: {service_role_key}")
    print(f"ANON_KEY: {anon_key}")