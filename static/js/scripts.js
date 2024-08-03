// Debounce function to delay execution
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Function to perform search
function performSearch() {
    const query = document.getElementById('query').value;
    const indexName = document.getElementById('index_name').value;
    const modelName = document.getElementById('model_name').value;
    const cutoff = document.getElementById('cutoff').value;

    // Send AJAX request to perform search
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `query=${query}&index_name=${indexName}&model_name=${modelName}&cutoff=${cutoff}`
    })
    .then(response => response.json())
    .then(data => {
        // Update search results on the page
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = '';

        if (data.search_results.length > 0) {
            data.search_results.forEach(result => {
                const resultItem = document.createElement('div');
                resultItem.className = 'grid-item';
                resultItem.innerHTML = `
                    <img src="${result.image_url}" alt="Product Image">
                    <p><strong>${result.short_descr}</strong></p>
                    <p>Price: ${result.minprice}</p>
                    <p>Manufacturer: ${result.manufacturer}</p>
                    <p>Similarity Score: ${result.similarity_score}</p>
                `;
                resultsContainer.appendChild(resultItem);
            });
        } else {
            resultsContainer.innerHTML = '<p>No results found.</p>';
        }
    })
    .catch(error => console.error('Error:', error));
}

// Add event listener for real-time search with debounce
document.addEventListener('DOMContentLoaded', function() {
    const debouncedSearch = debounce(performSearch, 250);  // 250ms = 0.25 seconds
    document.getElementById('query').addEventListener('input', debouncedSearch);
});
