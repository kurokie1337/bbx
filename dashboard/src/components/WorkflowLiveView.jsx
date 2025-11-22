/**
 * React component for live workflow execution view
 */
import React, { useState, useEffect } from 'react';

export function WorkflowLiveView({ workflowId }) {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState('connecting');
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Connect to WebSocket
    const websocket = new WebSocket(
      `ws://localhost:8000/ws/workflows/{}`
    );

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setStatus('connected');
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setEvents(prev => [...prev, data]);

      if (data.event === 'workflow_complete') {
        setStatus('completed');
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('error');
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus('disconnected');
    };

    // Ping every 30 seconds
    const pingInterval = setInterval(() => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    setWs(websocket);

    return () => {
      clearInterval(pingInterval);
      websocket.close();
    };
  }, [workflowId]);

  return (
    <div className="workflow-live-view">
      <h2>Workflow: {workflowId}</h2>
      <div className="status">Status: {status}</div>

      <div className="events-log">
        {events.map((event, idx) => (
          <div key={idx} className={`event event-{}`}>
            <span className="timestamp">{event.timestamp}</span>
            <span className="event-type">{event.event}</span>
            <span className="details">
              {JSON.stringify(event, null, 2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
