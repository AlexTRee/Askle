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

  useEffect(() => {
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

    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: question }),
      });

      const data = await response.json();

      // The /api/ask endpoint now returns a status and message.
      // You will need to call /api/status with the query to get the answer and references.
      console.log("/api/ask response:", data);
      setAnswer(data.message || 'Request submitted for processing.'); // Update UI based on the response
      // You should now implement a mechanism to call /api/status to get the actual answer and references.
      // For example, you could set an interval to poll the status endpoint.
      // For this immediate fix, we will not try to set answer and references from /api/ask directly.

    } catch (error) {
      console.error('Error asking question:', error);
      setAnswer('Sorry, there was an error submitting your question. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container my-5">
      <div className="row">
        <div className="col-12 text-center mb-4">
          <h1>Lung Cancer Research Assistant</h1>
          <p className="lead">Ask questions about the latest lung cancer research</p>
        </div>
      </div>

      <div className="row justify-content-center mb-5">
        <div className="col-md-8">
          <div className="card">
            <div className="card-body">
              <div className="mb-3">
                <label htmlFor="questionInput" className="form-label">Your Question:</label>
                <div className="input-group">
                  <input
                    type="text"
                    className="form-control"
                    id="questionInput"
                    placeholder="e.g., What are the latest advancements in immunotherapy for lung cancer?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
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
          </div>
        </div>
      </div>

      {answer && (
        <div className="row">
          <div className="col-12">
            <div className="card mb-4">
              <div className="card-header bg-primary text-white">
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

      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-header bg-secondary text-white">
              <h5 className="card-title mb-0">Recent Papers</h5>
            </div>
            <div className="card-body">
              <div className="papers-list">
                {papersLoading ? (
                  <div className="text-center py-4">
                    <div className="spinner-border" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                    <p className="mt-2">Loading recent papers...</p>
                  </div>
                ) : (
                  recentPapers.length > 0 ? (
                    recentPapers.map((paper, index) => (
                      <div key={index} className="paper-item mb-3">
                        <h6>
                          <a href={paper.url} target="_blank" rel="noreferrer">{paper.title}</a>
                        </h6>
                        <p className="mb-1">
                          <small>{paper.authors.join(', ')} - {paper.journal} ({paper.publication_date})</small>
                        </p>
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