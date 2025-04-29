import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import os
from dotenv import load_dotenv

class EmailAlertSystem:
    def __init__(self):
        load_dotenv()
        # Fixed configuration for your email
        self.config = {
            'sender_email': "mikekariuki10028@gmail.com",
            'receiver_email': "mikekariuki10028@gmail.com",
            'smtp_server': "smtp.gmail.com",
            'smtp_port': 465,
            'sender_password': os.getenv("GMAIL_APP_PASSWORD")
        }
        
    def update_config(self, sender=None, password=None, receiver=None):
        """Update configuration if needed"""
        if sender:
            self.config['sender_email'] = sender
        if password:
            self.config['sender_password'] = password
        if receiver:
            self.config['receiver_email'] = receiver
    
    def send_alert(self, subject, message):
        """Send email alert"""
        try:
            if not self.config['sender_password']:
                raise ValueError("Email password not configured")
            
            msg = MIMEMultipart()
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['receiver_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                self.config['smtp_server'],
                self.config['smtp_port'],
                context=context
            ) as server:
                server.login(
                    self.config['sender_email'],
                    self.config['sender_password']
                )
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Email sending failed: {str(e)}")
            return False