"""
Scheduler Agent for calendar and time management.

Handles:
- Calendar event management
- Intelligent scheduling
- Meeting preparation
- Reminder optimization
- Conflict resolution
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

from .base_agent import BaseAgent
from ..core.logger import logger


class SchedulerAgent(BaseAgent):
    """Agent specialized in calendar and scheduling operations."""

    # OAuth2 scopes for Google Calendar
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    def __init__(self, memory=None, llm=None):
        """Initialize scheduler agent."""
        super().__init__(
            name="scheduler",
            capabilities=[
                "create_event",
                "list_events",
                "update_event",
                "delete_event",
                "find_available_slots",
                "resolve_conflicts",
                "suggest_meeting_times"
            ],
            memory=memory,
            llm=llm
        )
        self.calendar_service = None
        self._initialize_calendar_service()

    def _initialize_calendar_service(self):
        """Initialize Google Calendar API service."""
        try:
            creds = None
            # Token file stores user's access and refresh tokens
            token_path = 'token.pickle'

            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)

            # If no valid credentials, let user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # For production, use environment variables
                    # This is a fallback for development
                    logger.warning("Calendar credentials not found - service will operate in limited mode")
                    return

            self.calendar_service = build('calendar', 'v3', credentials=creds)
            logger.info("Calendar service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize calendar service: {e}")
            self.calendar_service = None

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process scheduling task."""
        action = task.get("action", "")

        if not self.calendar_service:
            return {
                "success": False,
                "error": "Calendar service not initialized. Please configure OAuth credentials.",
                "suggestion": "Run calendar setup or use simulated mode for demo"
            }

        try:
            if action == "create_event":
                return await self._create_event(task)
            elif action == "list_events":
                return await self._list_events(task)
            elif action == "update_event":
                return await self._update_event(task)
            elif action == "delete_event":
                return await self._delete_event(task)
            elif action == "find_slots":
                return await self._find_available_slots(task)
            elif action == "resolve_conflicts":
                return await self._resolve_conflicts(task)
            elif action == "suggest_times":
                return await self._suggest_meeting_times(task)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _create_event(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a calendar event."""
        try:
            event_data = task.get("event", {})

            event = {
                'summary': event_data.get('title', 'Untitled Event'),
                'description': event_data.get('description', ''),
                'start': {
                    'dateTime': event_data.get('start_time'),
                    'timeZone': event_data.get('timezone', 'UTC'),
                },
                'end': {
                    'dateTime': event_data.get('end_time'),
                    'timeZone': event_data.get('timezone', 'UTC'),
                },
                'attendees': [
                    {'email': email} for email in event_data.get('attendees', [])
                ],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }

            if event_data.get('location'):
                event['location'] = event_data['location']

            created_event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()

            logger.info(f"Event created: {created_event.get('htmlLink')}")

            return {
                "success": True,
                "event_id": created_event['id'],
                "event_link": created_event.get('htmlLink'),
                "message": f"Event '{event['summary']}' created successfully"
            }

        except HttpError as e:
            logger.error(f"Calendar API error: {e}")
            return {
                "success": False,
                "error": f"API error: {str(e)}"
            }

    async def _list_events(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """List upcoming calendar events."""
        try:
            now = datetime.utcnow().isoformat() + 'Z'
            days_ahead = task.get('days_ahead', 7)
            max_results = task.get('max_results', 10)

            time_max = (
                datetime.utcnow() + timedelta(days=days_ahead)
            ).isoformat() + 'Z'

            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            if not events:
                return {
                    "success": True,
                    "events": [],
                    "message": "No upcoming events found"
                }

            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                formatted_events.append({
                    'id': event['id'],
                    'title': event.get('summary', 'No title'),
                    'start': start,
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'attendees': [
                        att.get('email') for att in event.get('attendees', [])
                    ]
                })

            return {
                "success": True,
                "events": formatted_events,
                "count": len(formatted_events)
            }

        except HttpError as e:
            logger.error(f"Calendar API error: {e}")
            return {
                "success": False,
                "error": f"API error: {str(e)}"
            }

    async def _find_available_slots(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Find available time slots for scheduling."""
        try:
            # Get current events
            events_result = await self._list_events({
                'days_ahead': task.get('days_ahead', 7),
                'max_results': 100
            })

            if not events_result['success']:
                return events_result

            # Use LLM to analyze schedule and suggest slots
            events = events_result.get('events', [])
            duration = task.get('duration_minutes', 60)

            prompt = f"""
            Analyze the following calendar events and suggest 3-5 available time slots
            for a {duration}-minute meeting in the next {task.get('days_ahead', 7)} days.

            Current events:
            {events}

            Consider:
            - Working hours (9 AM - 6 PM)
            - Buffer time between meetings (15 minutes)
            - Lunch time (12 PM - 1 PM)

            Return suggestions in JSON format with date, start_time, end_time.
            """

            suggestions = await self.llm.generate(prompt)

            return {
                "success": True,
                "available_slots": suggestions,
                "duration_minutes": duration
            }

        except Exception as e:
            logger.error(f"Error finding slots: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _resolve_conflicts(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve scheduling conflicts using AI."""
        conflicts = task.get('conflicts', [])

        prompt = f"""
        Analyze these scheduling conflicts and suggest resolutions:
        {conflicts}

        Consider:
        - Meeting priority and importance
        - Attendee availability
        - Meeting duration flexibility
        - Rescheduling impact

        Provide ranked recommendations for conflict resolution.
        """

        resolution = await self.llm.generate(prompt)

        return {
            "success": True,
            "resolution_suggestions": resolution,
            "conflicts_analyzed": len(conflicts)
        }

    async def _suggest_meeting_times(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal meeting times based on context."""
        participants = task.get('participants', [])
        duration = task.get('duration_minutes', 60)
        preferences = task.get('preferences', {})

        # Get calendar data
        events_result = await self._list_events({'days_ahead': 14})

        prompt = f"""
        Suggest optimal meeting times for the following:
        - Participants: {participants}
        - Duration: {duration} minutes
        - Preferences: {preferences}
        - Current schedule: {events_result.get('events', [])}

        Consider time zones, working hours, and meeting fatigue.
        Provide 3 best options with reasoning.
        """

        suggestions = await self.llm.generate(prompt)

        return {
            "success": True,
            "meeting_suggestions": suggestions,
            "participants_count": len(participants)
        }

    async def _update_event(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing calendar event."""
        try:
            event_id = task.get('event_id')
            updates = task.get('updates', {})

            # Get existing event
            event = self.calendar_service.events().get(
                calendarId='primary',
                eventId=event_id
            ).execute()

            # Apply updates
            for key, value in updates.items():
                if key in ['title', 'summary']:
                    event['summary'] = value
                elif key == 'description':
                    event['description'] = value
                elif key == 'start_time':
                    event['start']['dateTime'] = value
                elif key == 'end_time':
                    event['end']['dateTime'] = value

            updated_event = self.calendar_service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event
            ).execute()

            return {
                "success": True,
                "event_id": updated_event['id'],
                "message": "Event updated successfully"
            }

        except HttpError as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _delete_event(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a calendar event."""
        try:
            event_id = task.get('event_id')

            self.calendar_service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()

            return {
                "success": True,
                "message": f"Event {event_id} deleted successfully"
            }

        except HttpError as e:
            return {
                "success": False,
                "error": str(e)
            }
