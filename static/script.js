// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Load recent papers on page load
    loadRecentPapers();
    
    // Set up question submission
    const askButton = document.getElementById('askButton');
    const questionInput = document.getElementById('questionInput');
    
    askButton.addEventListener('click', function() {
        askQuestion();
    });
    
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            askQuestion();
        }
    });
});

function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) {
        alert('Please enter a question');
        return;
    }
    
    // Show loading spinner
    showSpinner('Analyzing research papers...');
    
    // Make API request
    fetch('/api/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Hide spinner
        hideSpinner();
        
        // Display answer
        document.getElementById('answerCard').classList.remove('d-none');
        document.getElementById('answerContent').innerHTML = data.answer;
        
        // Display references
        const referencesContainer = document.getElementById('references');
        referencesContainer.innerHTML = '';
        
        data.references.forEach((paper, index) => {
            const paperCard = createPaperCard(paper, index + 1);
            referencesContainer.appendChild(paperCard);
        });
        
        // Scroll to answer
        document.getElementById('answerCard').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        hideSpinner();
        alert('Error: ' + error.message);
    });
}

function loadRecentPapers() {
    fetch('/api/recent')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(papers => {
        const recentPapersContainer = document.getElementById('recentPapers');
        recentPapersContainer.innerHTML = '';
        
        if (papers.length === 0) {
            recentPapersContainer.innerHTML = '<p class="text-center">No recent papers found. They will be fetched soon.</p>';
            return;
        }
        
        papers.forEach((paper, index) => {
            const paperCard = createPaperCard(paper);
            recentPapersContainer.appendChild(paperCard);
        });
    })
    .catch(error => {
        const recentPapersContainer = document.getElementById('recentPapers');
        recentPapersContainer.innerHTML = '<p class="text-center text-danger">Error loading papers: ' + error.message + '</p>';
    });
}

function createPaperCard(paper, referenceNum = null) {
    const card = document.createElement('div');
    card.className = 'card paper-card';
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    // Create reference number if provided
    let titlePrefix = '';
    if (referenceNum !== null) {
        titlePrefix = `[${referenceNum}] `;
    }
    
    // Create title with link
    const title = document.createElement('h5');
    title.className = 'paper-title';
    const titleLink = document.createElement('a');
    titleLink.href = paper.url;
    titleLink.target = '_blank';
    titleLink.textContent = titlePrefix + paper.title;
    title.appendChild(titleLink);
    
    // Create authors and metadata
    const metadata = document.createElement('p');
    metadata.className = 'paper-authors';
    metadata.textContent = `${paper.authors} | ${paper.publication_date || 'Unknown date'} | Source: ${paper.source}`;
    
    // Create summary
    const summary = document.createElement('p');
    summary.className = 'paper-summary';
    summary.textContent = paper.summary || (paper.abstract ? paper.abstract.substring(0, 200) + '...' : 'No summary available');
    
    // Assemble card
    cardBody.appendChild(title);
    cardBody.appendChild(metadata);
    cardBody.appendChild(summary);
    card.appendChild(cardBody);
    
    return card;
}

function showSpinner(message) {
    // Create spinner overlay if it doesn't exist
    if (!document.querySelector('.spinner-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'spinner-overlay';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner-border text-primary';
        spinner.setAttribute('role', 'status');
        
        const spinnerText = document.createElement('span');
        spinnerText.className = 'visually-hidden';
        spinnerText.textContent = 'Loading...';
        
        const messageElement = document.createElement('p');
        messageElement.className = 'mt-3 spinner-message';
        messageElement.textContent = message || 'Loading...';
        
        spinner.appendChild(spinnerText);
        overlay.appendChild(spinner);
        overlay.appendChild(messageElement);
        
        document.body.appendChild(overlay);
    } else {
        document.querySelector('.spinner-message').textContent = message || 'Loading...';
        document.querySelector('.spinner-overlay').style.display = 'flex';
    }
}

function hideSpinner() {
    const overlay = document.querySelector('.spinner-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}