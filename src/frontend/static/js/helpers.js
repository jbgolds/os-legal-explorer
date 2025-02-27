// Format date for display
function format_date(dateString) {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Get color class based on treatment
function getTreatmentColor(treatment) {
    switch (treatment) {
        case 'POSITIVE': return 'bg-green-100 text-green-800';
        case 'NEGATIVE': return 'bg-red-100 text-red-800';
        case 'CAUTION': return 'bg-yellow-100 text-yellow-800';
        default: return 'bg-gray-100 text-gray-800';
    }
}
