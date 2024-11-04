# ABI Code with Streamlit
Run Script:
python st_run_thread.py   

Kill Script:

pkill -f streamlit

File size shows 1Gig, even though I have 512 configured
(abi) ubuntu@ip-10-10-1-63:~/.streamlit$ cat config.toml 
[server]
maxUploadSize = 512


#################
### Set the maximum upload size for Streamlit (e.g., 512 MB)
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '512'  # MB, default is 200MB


## nginx proxy:

### Creating nginx proxy and ssl certificate

To enable HTTPS on a Streamlit server using a self-signed certificate, you will first need to create the certificate and then configure Streamlit to use it. Using a self-signed certificate is useful for development or internal applications but it's not recommended for production use due to security limitations and because users will see a browser warning about the certificate not being trusted.
Here's how you can generate a self-signed certificate and use it with Streamlit:
Step 1: Generate Self-Signed SSL Certificate
You can use OpenSSL to generate a self-signed SSL certificate. Open your terminal and follow these steps:
1. Navigate to a directory where you want to store your certificates, or create a new one.
2. Generate a private key and certificate:

mkdir sslcert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem


This command does the following:
	• req: Command to manage certificate signing requests.
	• -x509: Specifies that you want to create a self-signed certificate.
	• -nodes: Skips the option to secure the certificate with a passphrase. This is useful for automated processes.
	• -days 365: Sets the certificate to expire after one year. You can change this value as needed.
	• -newkey rsa:2048: Creates a new certificate request and a new private key. rsa:2048 specifies a 2048-bit RSA key.
	• -keyout: Specifies the name of the output file for the private key.
	• -out: Specifies the name of the output file for the certificate.

3. Fill in the prompts for your country code, state, locality, organization name, organizational unit, common name (your IP address or domain name), and email address. Since this is a self-signed certificate, the information doesn't need to be validated by a third party, but it should be relevant to your application or organization.


Step 2: Configure Streamlit to Use HTTPS
Streamlit doesn’t natively support HTTPS directly through its settings, so you typically would run it behind a reverse proxy that handles HTTPS. However, for development purposes, you can use a workaround like running Streamlit through nginx or a similar web server that can use the SSL certificate, or using tools like stunnel.
Using nginx as a Reverse Proxy
Here’s a basic setup for nginx to use as a reverse proxy for Streamlit:

sudo apt update
sudo apt install nginx

Configure nginx:
	• Edit the nginx configuration file (often found in /etc/nginx/sites-available/default or create a new file in the sites-available directory).
	• Configure it to proxy requests to your Streamlit app and use the SSL certificate:

Configure NGINX Proxy:

sudo vi /etc/nginx/sites-available/default

server {
    listen 443 ssl;
    #server_name your_domain_or_ip;
    server_name 23.20.149.111;

    ssl_certificate /home/ubuntu/sslcert/mycert.pem;
    ssl_certificate_key /home/ubuntu/sslcert/mykey.key;

    client_max_body_size 200M; # allows file upload for 200 mbytes

    location / {
        proxy_pass http://localhost:8501; # Default Streamlit port
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

Test the configuration and then reload nginx
sudo nginx -t
sudo systemctl reload nginx


## Increase file upload limit
```
(abi) ubuntu@ip-10-10-1-123:~$ pwd
/home/ubuntu

mkdir -p .streamlit
echo "[server]\nmaxUploadSize = 500" > .streamlit/config.toml
```
```
(abi) ubuntu@ip-10-10-1-123:~$ pwd
/home/ubuntu
(abi) ubuntu@ip-10-10-1-123:~$ cat .streamlit/config.toml 
[server]
maxUploadSize = 512




