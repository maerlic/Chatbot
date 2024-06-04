import React, { useReducer, useEffect, useState } from 'react';
import axios from 'axios';
import './ChatWindow.css'; // Ensure this import is at the top of your file

const initialState = {
  messages: []
};

function reducer(state, action) {
  switch (action.type) {
    case 'add_message':
      return { messages: [...state.messages, action.payload] };
    case 'add_response':
      return { messages: [...state.messages, action.payload] };
    default:
      throw new Error();
  }
}

const ChatWindow = () => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input) return;

    const newMessage = { role: 'user', content: input };
    dispatch({ type: 'add_message', payload: newMessage });
    console.log('Sending message:', newMessage);

    try {
      const response = await axios.post('http://localhost:5000/chat', { message: input });
      console.log('Response received:', response.data);
      dispatch({ type: 'add_response', payload: response.data });
    } catch (error) {
      console.error('Error sending message:', error.toJSON ? error.toJSON() : error);
      dispatch({ type: 'add_response', payload: { role: 'assistant', content: 'Error sending message' } });
    }

    setInput('');
  };

  useEffect(() => {
    const chatWindow = document.getElementById('chat-window');
    if (chatWindow) {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  }, [state.messages]);

  return (
    <div className="chat-container">
      <div id="chat-window" className="chat-window">
        {state.messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.role}`}>
            <p>{msg.content}</p>
          </div>
        ))}
      </div>
      <div className="chat-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          className="chat-input"
        />
        <button onClick={sendMessage} className="chat-send-button">
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;
