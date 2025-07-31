// src/App.js
import React, { useState } from 'react';
import './App.css';

/**
 * Main application component for the PDF Chatbot.
 * Handles PDF uploads and chat interactions with a FastAPI backend.
 * @returns {JSX.Element} The rendered React component.
 */
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [pdfId, setPdfId] = useState(null);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');
  const [chatStatus, setChatStatus] = useState('');

  /**
   * Handles the change event for the file input.
   * @param {Event} event The DOM event object.
   */
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  /**
   * Handles the PDF upload process to the backend.
   * Sends the selected file to the /upload-pdf/ endpoint.
   */
  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setUploadStatus('Uploading and processing PDF...');
      const response = await fetch('http://localhost:8000/upload-pdf/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setPdfId(data.pdf_id);
        setUploadStatus('PDF uploaded and processed successfully! You can now chat.');
        setChatHistory([]); // Clear chat history for new PDF
      } else {
        setUploadStatus(`Error: ${data.detail || 'Something went wrong.'}`);
      }
    } catch (error) {
      setUploadStatus(`Network error: ${error.message}`);
    }
  };

  /**
   * Handles sending a chat message to the backend.
   * Sends the user's message and the current PDF ID to the /chat/ endpoint.
   */
  const handleChat = async () => {
    if (!chatMessage.trim()) {
      setChatStatus('Please enter a message.');
      return;
    }
    if (!pdfId) {
      setChatStatus('Please upload a PDF first.');
      return;
    }

    const userMessage = { sender: 'user', text: chatMessage };
    setChatHistory((prev) => [...prev, userMessage]);
    setChatMessage('');
    setChatStatus('Getting response...');

    try {
      const response = await fetch('http://localhost:8000/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: chatMessage, pdf_id: pdfId }),
      });
      const data = await response.json();
      if (response.ok) {
        const aiMessage = { sender: 'ai', text: data.response };
        setChatHistory((prev) => [...prev, aiMessage]);
        setChatStatus('');
      } else {
        setChatStatus(`Error: ${data.detail || 'Something went wrong.'}`);
      }
    } catch (error) {
      setChatStatus(`Network error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <h1>PDF Chatbot</h1>

      <div className="upload-section">
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!selectedFile}>Upload PDF</button>
        <p className="status">{uploadStatus}</p>
      </div>

      {pdfId && (
        <div className="chat-section">
          <h2>Chat with PDF (ID: {pdfId})</h2>
          <div className="chat-history">
            {chatHistory.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <strong>{msg.sender === 'user' ? 'You' : 'AI'}:</strong> {msg.text}
              </div>
            ))}
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              placeholder="Ask a question about the PDF..."
            />
            <button onClick={handleChat} disabled={!chatMessage.trim()}>Send</button>
          </div>
          <p className="status">{chatStatus}</p>
        </div>
      )}
    </div>
  );
}

export default App;
