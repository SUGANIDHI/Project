import os
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Load .env file from the same directory as this script
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


def send_email(recipient: str, subject: str, body: str, attachments: list = None) -> bool:
    """Send an email with optional image attachments using SMTP.

    Args:
        recipient: Email address to send to
        subject: Email subject line
        body: Email body text
        attachments: Optional list of file paths to attach

    The SMTP server configuration is read from environment variables:
        EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, EMAIL_USE_TLS
    Returns True on success, False otherwise.
    """
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", "587"))
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    use_tls = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"

    if not all([host, user, password]):
        print(f"SMTP configuration missing: host={host}, user={user}, password={'***' if password else None}")
        return False

    # Create multipart message for attachments
    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = recipient
    msg["Subject"] = subject
    
    # Add body text
    msg.attach(MIMEText(body, "plain"))
    
    # Add attachments if provided
    if attachments:
        for file_path in attachments:
            path = Path(file_path)
            if path.exists() and path.is_file():
                with open(path, "rb") as f:
                    img_data = f.read()
                    img = MIMEImage(img_data, name=path.name)
                    img.add_header("Content-Disposition", "attachment", filename=path.name)
                    msg.attach(img)
                print(f"Attached: {path.name}")
            else:
                print(f"Attachment not found: {file_path}")

    try:
        with smtplib.SMTP(host, port) as server:
            if use_tls:
                server.starttls()
            server.login(user, password)
            server.send_message(msg)
        print(f"Email sent to {recipient} with {len(attachments) if attachments else 0} attachments")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
