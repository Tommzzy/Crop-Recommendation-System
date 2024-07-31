document.getElementById('crop-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => (data[key] = value));

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<p>Loading...</p>'; // Show loading message

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(result => {
        resultsDiv.innerHTML = '<h2>Recommended Crops</h2>';
        if (result.predicted_crops.length > 0) {
            const ul = document.createElement('ul');
            result.predicted_crops.forEach(crop => {
                const li = document.createElement('li');
                li.innerHTML = `<img src="/static/images/${crop.image}" alt="${crop.crop}" style="width: 50px; height: 50px; margin-right: 10px;"> ${crop.crop}: ${(crop.probability * 100).toFixed(2)}%`;
                ul.appendChild(li);
            });
            resultsDiv.appendChild(ul);
        } else {
            resultsDiv.innerHTML += '<p>No suitable crops found.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultsDiv.innerHTML = '<p>There was an error processing your request. Please try again later.</p>';
    });
});
