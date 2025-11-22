"""Event store implementation."""

import json
from datetime import datetime
from typing import List

import psycopg2


class Event:
    def __init__(self, aggregate_id, event_type, data, timestamp=None):
        self.aggregate_id = aggregate_id
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()


class EventStore:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        self._init_table()

    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    aggregate_id VARCHAR(255),
                    event_type VARCHAR(255),
                    data JSONB,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """
            )
        self.conn.commit()

    def append(self, event: Event):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (aggregate_id, event_type, data, timestamp) VALUES (%s, %s, %s, %s)",
                (
                    event.aggregate_id,
                    event.event_type,
                    json.dumps(event.data),
                    event.timestamp,
                ),
            )
        self.conn.commit()

    def get_events(self, aggregate_id: str) -> List[Event]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT event_type, data, timestamp FROM events WHERE aggregate_id = %s ORDER BY id",
                (aggregate_id,),
            )
            return [
                Event(aggregate_id, row[0], json.loads(row[1]), row[2])
                for row in cur.fetchall()
            ]
