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
                resultItem.dataset.shortDescr = result.short_descr; // Add dataset for the short description
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
    const debouncedSearch = debounce(performSearch, 500);  // 500ms = 0.5 seconds
    document.getElementById('query').addEventListener('input', debouncedSearch);
});

// Function to open the modal
function openModal(shortDescr, indexName, modelName, productId) {
    const modal = document.getElementById('relatedModal');
    const relatedProductsContainer = document.getElementById('related-products-container');
    const searchQuery = document.getElementById('query').value; // Get the current search query
    
    // Set the modal header to the product's short description
    document.getElementById('modal-header').textContent = shortDescr;
    
    // Log the click event with the search query
    logEvent('Click', { shortDescr, productId }, searchQuery);

    modal.style.display = 'block';

    // Fetch related products using AJAX
    fetch(`/related-products?short_descr=${encodeURIComponent(shortDescr)}&index_name=${indexName}&model_name=${modelName}`)
        .then(response => response.json())
        .then(data => {
            // Clear existing content
            relatedProductsContainer.innerHTML = '';

            if (data.related_products) {
                // Populate related products and log events for unclicked products
                data.related_products.forEach(product => {
                    const productItem = document.createElement('div');
                    productItem.className = 'grid-item';
                    productItem.innerHTML = `
                        <img src="${product.image_url}" alt="Product Image">
                        <p><strong>${product.short_descr}</strong></p>
                        <p>Price: ${product.minprice}</p>
                        <p>Manufacturer: ${product.manufacturer}</p>
                    `;
                    relatedProductsContainer.appendChild(productItem);

                    // Log the unclicked products with the search query
                    if (product.productId !== productId) {
                        logEvent('Not Clicked', product, searchQuery);
                    }
                });
            } else {
                relatedProductsContainer.innerHTML = '<p>No related products found.</p>';
            }
        })
        .catch(error => console.error('Error:', error));
}


// Function to close the modal
function closeModal() {
    const modal = document.getElementById('relatedModal');
    modal.style.display = 'none';
}

// Event listener for product cards
document.addEventListener('click', function(e) {
    if (e.target.closest('.grid-item')) {
        const shortDescr = e.target.closest('.grid-item').dataset.shortDescr;
        const indexName = document.getElementById('index_name').value;
        const modelName = document.getElementById('model_name').value;
        openModal(shortDescr, indexName, modelName);
    }
});

// Event listener for the close button
document.querySelector('.close').addEventListener('click', closeModal);

// Event listener for "Add to Basket" button
document.getElementById('add-to-basket-btn').addEventListener('click', function() {
    const modalHeader = document.getElementById('modal-header').textContent;
    const searchQuery = document.getElementById('query').value; // Get the current search query
    logEvent('Add to Basket', { shortDescr: modalHeader }, searchQuery);
    alert('Product added to basket!');
});

// Event listener to close the modal when clicking outside of it
window.addEventListener('click', function(event) {
    const modal = document.getElementById('relatedModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
});

// Function to log events
function logEvent(eventType, productInfo, searchQuery) {
    fetch('/log-event', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            event_type: eventType,
            product_info: productInfo,
            search_query: searchQuery
        })
    })
    .catch(error => console.error('Error logging event:', error));
}