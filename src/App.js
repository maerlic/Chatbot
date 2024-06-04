// src/App.js
import React from 'react';
import ChatWindow from './components/ChatWindow.js'; // Add the .js extension
import './App.css'; // Import the CSS file for styling

function App() {
  return (
    <div className="App">
      <div className="chat-container-wrapper">
        <ChatWindow />
      </div>
    </div>
  );
}

export default App;
