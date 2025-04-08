// src/App.js
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [references, setReferences] = useState([]);
  const [recentPapers, setRecentPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [papersLoading, setPapersLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false); // State to control fade-in

  useEffect(() => {
    // Simulate initial load
    setTimeout(() => {
      setHasLoaded(true);
    }, 300); // Adjust delay as needed

    // Fetch recent papers when component mounts
    fetchRecentPapers();
  }, []);

  const fetchRecentPapers = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/papers/recent');
      const data = await response.json();
      setRecentPapers(data);
      setPapersLoading(false);
    } catch (error) {
      console.error('Error fetching recent papers:', error);
      setPapersLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer('');
    setReferences([]);
    setProcessing(false); // Reset processing state

    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question }),
      });

      const data = await response.json();
      console.log("/api/ask response:", data);
      setAnswer(data.message || 'Request submitted for processing.'); // Update UI based on the response
      setLoading(false);
      setProcessing(true); // Indicate processing has started
      // Start polling for status
      startStatusPolling(question);
    } catch (error) {
      console.error('Error asking question:', error);
      setAnswer('Sorry, there was an error submitting your question. Please try again.');
      setLoading(false);
      setProcessing(false);
    }
  };

  const startStatusPolling = (query) => {
    const intervalId = setInterval(async () => {
      try {
        const statusResponse = await fetch(`http://localhost:8000/api/status?query=${encodeURIComponent(query)}`);
        const statusData = await statusResponse.json();
        console.log("/api/status response:", statusData);

        if (statusData.status === 'completed') {
          setAnswer(statusData.summaries[0]?.summary || 'No summary generated.'); // Adjust based on your desired output
          setReferences(statusData.summaries.map(s => ({
            title: s.title,
            url: s.url,
            authors: s.authors,
            journal: s.journal,
            publication_date: s.publication_date,
          })));
          setProcessing(false);
          clearInterval(intervalId);
        } else if (statusData.status === 'error') {
          setAnswer(`Error processing your question: ${statusData.error_message || 'Unknown error.'}`);
          setProcessing(false);
          clearInterval(intervalId);
        }
        // If status is 'processing', continue polling

      } catch (error) {
        console.error('Error checking status:', error);
        setAnswer('Sorry, there was an error retrieving the answer. Please try again later.');
        setProcessing(false);
        clearInterval(intervalId);
      }
    }, 3000); // Poll every 3 seconds (adjust as needed)

    // Adding a timeout to stop polling after a certain duration
    setTimeout(() => {
      if (processing) {
        setAnswer('Processing taking too long. Please try again later.');
        setProcessing(false);
        clearInterval(intervalId);
      }
    }, 120000); // Timeout after 120 seconds
  };

  return (
    <div className={`container my-5 ${hasLoaded ? 'fade-in' : ''}`}>
      <div className="row">
        <div className="col-12 text-center hero">
          <h1>Askle Cancer Research Assistant</h1>
          <p className="lead">Ask questions about the latest medical research</p>
        </div>
      </div>

      <div className="row justify-content-center mb-5 question-section">
        <div className="col-md-8">
          <label htmlFor="questionInput" className="form-label">Your Question:</label>
          <div className="input-group">
            <input
              type="text"
              className="form-control"
              id="questionInput"
              placeholder="e.g., What are the latest advancements in immunotherapy for lung cancer?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAskQuestion()}
            />
            <button
              className="btn btn-primary"
              onClick={handleAskQuestion}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                  Processing...
                </>
              ) : 'Ask'}
            </button>
          </div>
        </div>
      </div>

      {answer && (
        <div className="row answer-section">
          <div className="col-12">
            <div className="card mb-4">
              <div className="card-header">
                <h5 className="card-title mb-0">Answer</h5>
              </div>
              <div className="card-body">
                <div className="mb-3">{answer}</div>
                {references.length > 0 && (
                  <>
                    <hr />
                    <h6>References:</h6>
                    <div className="references-list">
                      {references.map((ref, index) => (
                        <div key={index} className="reference-item">
                          <a href={ref.url} target="_blank" rel="noreferrer">{ref.title}</a>
                          <p><small>{ref.authors.join(', ')} - {ref.journal} ({ref.publication_date})</small></p>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="row recent-papers-section">
        <div className="col-12">
          <div className="card">
            <div className="card-header bg-secondary text-white">
              <h5 className="card-title mb-0">Recent Papers</h5>
            </div>
            <div className="card-body">
              <div className="papers-list">
                {papersLoading ? (
                  <div className="loading-spinner">
                    <div className="spinner-border" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                    <p>Loading recent papers...</p>
                  </div>
                ) : (
                  recentPapers.length > 0 ? (
                    recentPapers.map((paper, index) => (
                      <div key={index} className="paper-item mb-3">
                        <h6>
                          <a href={paper.url} target="_blank" rel="noreferrer">{paper.title}</a>
                        </h6>
                        <p className="mb-1">
                          <small>{paper.authors.join(', ')} - {paper.journal} ({paper.publication_date})</small></p>
                        <p className="paper-abstract">{paper.abstract.substring(0, 200)}...</p>
                      </div>
                    ))
                  ) : (
                    <p>No recent papers available.</p>
                  )
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;