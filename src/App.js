// src/App.js
import React, { useState, useEffect } from 'react';
import { Container, Box, Typography, TextField, Button, CircularProgress, Card, CardContent, Divider, Link } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ArticleIcon from '@mui/icons-material/Article';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [recentQueries, setRecentQueries] = useState([]);
  
  // Fetch recent queries on load
  useEffect(() => {
    fetchRecentQueries();
  }, []);
  
  const fetchRecentQueries = async () => {
    try {
      const response = await fetch('/api/history');
      if (response.ok) {
        const data = await response.json();
        setRecentQueries(data);
      }
    } catch (err) {
      console.error("Failed to fetch recent queries:", err);
    }
  };
  
  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a search query.');
      return;
    }
    
    setIsLoading(true);
    setError('');
    setResults([]);
    
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          max_results: 10
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }
      
      // Start polling for results
      const initialData = await response.json();
      
      if (initialData.summaries.length > 0) {
        // Results are already available
        setResults(initialData.summaries);
        setIsLoading(false);
      } else {
        // Poll for results
        pollForResults(query);
      }
      
    } catch (err) {
      setError('An error occurred while searching. Please try again.');
      setIsLoading(false);
      console.error('Search error:', err);
    }
  };
  
  const pollForResults = async (searchQuery) => {
    try {
      // In a real application, you would use WebSockets instead of polling
      const interval = setInterval(async () => {
        const response = await fetch(`/api/status?query=${encodeURIComponent(searchQuery)}`);
        if (response.ok) {
          const data = await response.json();
          
          if (data.status === 'completed') {
            setResults(data.summaries);
            setIsLoading(false);
            clearInterval(interval);
            
            // Update recent queries
            fetchRecentQueries();
          }
        }
      }, 2000);
      
      // Set a timeout to stop polling after 30 seconds
      setTimeout(() => {
        clearInterval(interval);
        if (isLoading) {
          setIsLoading(false);
          setError('Request timed out. Please try again.');
        }
      }, 30000);
      
    } catch (err) {
      setError('An error occurred while retrieving results.');
      setIsLoading(false);
    }
  };
  
  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Lung Cancer Research Assistant
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 4 }}>
          Ask questions about the latest lung cancer research
        </Typography>
        
        <Box component="form" onSubmit={handleSearch} sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <TextField
              fullWidth
              label="Ask a question about lung cancer research"
              variant="outlined"
              value={query}
              onChange={e => setQuery(e.target.value)}
              sx={{ maxWidth: 600, mr: 1 }}
            />
            <Button 
              type="submit" 
              variant="contained" 
              disabled={isLoading}
              startIcon={<SearchIcon />}
            >
              Search
            </Button>
          </Box>
          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}
        </Box>
        
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>
              Searching for the latest research...
            </Typography>
          </Box>
        ) : (
          <>
            {results.length > 0 && (
              <Box>
                <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 3 }}>
                  Research Results
                </Typography>
                
                {results.map((paper, index) => (
                  <Card key={index} sx={{ mb: 3, textAlign: 'left' }}>
                    <CardContent>
                      <Typography variant="h6" component="h3" gutterBottom>
                        {paper.title}
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {paper.authors.join(', ')} - {paper.publication_date} - {paper.journal}
                      </Typography>
                      
                      <Typography variant="body2" color="text.primary" paragraph sx={{ mt: 2 }}>
                        <strong>Summary:</strong> {paper.summary}
                      </Typography>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="caption" color="text.secondary">
                          Source: {paper.source}
                        </Typography>
                        
                        <Link 
                          href={paper.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          sx={{ display: 'flex', alignItems: 'center' }}
                        >
                          <ArticleIcon sx={{ mr: 0.5, fontSize: 16 }} />
                          View Original Paper
                        </Link>
                      </Box>
                    </CardContent>
                  </Card>
                ))}
              </Box>
            )}
            
            {results.length === 0 && !isLoading && (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 4 }}>
                <InfoOutlinedIcon color="primary" sx={{ fontSize: 60, mb: 2 }} />
                <Typography variant="body1">
                  Ask a question about lung cancer research to get started.
                </Typography>
                
                {recentQueries.length > 0 && (
                  <Box sx={{ mt: 4, width: '100%', maxWidth: 600 }}>
                    <Typography variant="h6" gutterBottom>
                      Recent Searches
                    </Typography>
                    {recentQueries.slice(0, 5).map((item, index) => (
                      <Card key={index} sx={{ mb: 1, cursor: 'pointer' }} onClick={() => setQuery(item.query)}>
                        <CardContent sx={{ py: 1 }}>
                          <Typography variant="body2">{item.query}</Typography>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                )}
              </Box>
            )}
          </>
        )}
      </Box>
    </Container>
  );
}

export default App;