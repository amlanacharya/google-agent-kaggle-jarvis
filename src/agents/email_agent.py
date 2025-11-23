"""
Email Agent for Gmail integration.

Handles:
- Email reading and search
- Smart categorization
- Auto-draft generation
- Follow-up tracking
- Attachment processing
"""

import os
import base64
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

from .base_agent import BaseAgent
from ..core.logger import logger


class EmailAgent(BaseAgent):
    """Agent specialized in email management."""

    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify'
    ]

    def __init__(self, memory=None, llm=None):
        """Initialize email agent."""
        super().__init__(
            name="email",
            capabilities=[
                "read_emails",
                "send_email",
                "draft_email",
                "categorize_emails",
                "search_emails",
                "summarize_thread",
                "extract_action_items",
                "smart_reply"
            ],
            memory=memory,
            llm=llm
        )
        self.gmail_service = None
        self._initialize_gmail_service()

    def _initialize_gmail_service(self):
        """Initialize Gmail API service."""
        try:
            creds = None
            token_path = 'gmail_token.pickle'

            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    logger.warning("Gmail credentials not found - service will operate in limited mode")
                    return

            self.gmail_service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gmail service: {e}")
            self.gmail_service = None

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process email task."""
        action = task.get("action", "")

        if not self.gmail_service:
            return {
                "success": False,
                "error": "Gmail service not initialized. Please configure OAuth credentials.",
                "suggestion": "Run Gmail setup or use simulated mode for demo"
            }

        try:
            if action == "read_emails":
                return await self._read_emails(task)
            elif action == "send_email":
                return await self._send_email(task)
            elif action == "draft_email":
                return await self._draft_email(task)
            elif action == "categorize":
                return await self._categorize_emails(task)
            elif action == "search":
                return await self._search_emails(task)
            elif action == "summarize_thread":
                return await self._summarize_thread(task)
            elif action == "extract_actions":
                return await self._extract_action_items(task)
            elif action == "smart_reply":
                return await self._smart_reply(task)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
        except Exception as e:
            logger.error(f"Email agent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _read_emails(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Read recent emails."""
        try:
            max_results = task.get('max_results', 10)
            query = task.get('query', 'is:unread')

            results = self.gmail_service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()

            messages = results.get('messages', [])

            if not messages:
                return {
                    "success": True,
                    "emails": [],
                    "message": "No messages found"
                }

            emails = []
            for msg in messages:
                msg_data = self.gmail_service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()

                headers = msg_data['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')

                # Get email body
                body = self._get_email_body(msg_data['payload'])

                emails.append({
                    'id': msg['id'],
                    'subject': subject,
                    'from': sender,
                    'date': date,
                    'snippet': msg_data.get('snippet', ''),
                    'body': body[:500]  # Truncate for preview
                })

            return {
                "success": True,
                "emails": emails,
                "count": len(emails)
            }

        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_email_body(self, payload):
        """Extract email body from payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    return base64.urlsafe_b64decode(data).decode('utf-8')
        elif 'body' in payload:
            data = payload['body'].get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8')
        return ""

    async def _send_email(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send an email."""
        try:
            to = task.get('to')
            subject = task.get('subject')
            body = task.get('body')

            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject

            raw_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')

            sent_message = self.gmail_service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()

            logger.info(f"Email sent: {sent_message['id']}")

            return {
                "success": True,
                "message_id": sent_message['id'],
                "message": f"Email sent to {to}"
            }

        except HttpError as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _draft_email(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate email draft using AI."""
        context = task.get('context', '')
        purpose = task.get('purpose', '')
        tone = task.get('tone', 'professional')

        prompt = f"""
        Generate an email draft with the following:

        Purpose: {purpose}
        Context: {context}
        Tone: {tone}

        Include:
        - Appropriate subject line
        - Well-structured body
        - Professional closing

        Format as JSON with 'subject' and 'body' fields.
        """

        draft = await self.llm.generate(prompt)

        return {
            "success": True,
            "draft": draft,
            "message": "Draft generated successfully"
        }

    async def _categorize_emails(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize emails using AI."""
        emails = task.get('emails', [])

        prompt = f"""
        Categorize the following emails into these categories:
        - Urgent: Requires immediate attention
        - Important: Should be handled today
        - Follow-up: Needs response but not urgent
        - Information: FYI only
        - Spam/Low priority: Can be ignored

        Emails:
        {emails}

        Return JSON with email IDs and assigned categories.
        """

        categorization = await self.llm.generate(prompt)

        return {
            "success": True,
            "categorization": categorization,
            "emails_processed": len(emails)
        }

    async def _search_emails(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Search emails with advanced query."""
        query = task.get('query', '')
        max_results = task.get('max_results', 20)

        return await self._read_emails({
            'query': query,
            'max_results': max_results
        })

    async def _summarize_thread(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize email thread."""
        thread_id = task.get('thread_id')

        try:
            thread = self.gmail_service.users().threads().get(
                userId='me',
                id=thread_id
            ).execute()

            messages = thread.get('messages', [])

            # Extract content from all messages
            thread_content = []
            for msg in messages:
                headers = msg['payload']['headers']
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                body = self._get_email_body(msg['payload'])

                thread_content.append({
                    'from': sender,
                    'body': body
                })

            prompt = f"""
            Summarize this email thread:
            {thread_content}

            Provide:
            1. Main topic/subject
            2. Key points discussed
            3. Action items (if any)
            4. Current status
            """

            summary = await self.llm.generate(prompt)

            return {
                "success": True,
                "summary": summary,
                "message_count": len(messages)
            }

        except HttpError as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _extract_action_items(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action items from emails."""
        emails = task.get('emails', [])

        prompt = f"""
        Extract action items from these emails:
        {emails}

        For each action item, provide:
        - Description
        - Deadline (if mentioned)
        - Priority (high/medium/low)
        - Assigned to (if mentioned)

        Return as structured JSON.
        """

        action_items = await self.llm.generate(prompt)

        return {
            "success": True,
            "action_items": action_items,
            "emails_analyzed": len(emails)
        }

    async def _smart_reply(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate smart reply suggestions."""
        email_content = task.get('email', '')

        prompt = f"""
        Generate 3 smart reply options for this email:
        {email_content}

        Provide:
        1. Quick acknowledgment
        2. Detailed response
        3. Defer/schedule option

        Each should be professional and contextually appropriate.
        """

        replies = await self.llm.generate(prompt)

        return {
            "success": True,
            "reply_suggestions": replies
        }
