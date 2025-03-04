document.addEventListener('DOMContentLoaded', function () {
    // Search input and button
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');

    // Reset buttons
    const resetButtons = document.querySelectorAll('.reset-filters');

    // Year picker elements
    const yearPicker = document.querySelector('.year-picker');
    const decadeDisplay = document.querySelector('.decade-display');
    const prevDecadeBtn = document.querySelector('.prev-decade-btn');
    const nextDecadeBtn = document.querySelector('.next-decade-btn');
    const yearsList = document.querySelector('.years-list');
    const resetYearBtn = document.querySelector('.reset-year-filters');
    const yearFromInput = document.querySelector('input[name="year_from"]');
    const yearToInput = document.querySelector('input[name="year_to"]');

    // Fix for dropdown positioning
    const searchBox = document.querySelector('.flex.w-full.border.rounded-lg');
    if (searchBox) {
        searchBox.style.overflow = 'visible';
    }

    // Court filter elements
    const allCourtsCheckbox = document.querySelector('input[value="all"]');
    const courtOptions = document.querySelectorAll('.court-option');

    // Initialize year picker
    initializeYearPicker();

    // Initialize court filter functionality
    initializeCourtFilter();

    // Event listeners for reset buttons
    resetButtons.forEach(button => {
        button.addEventListener('click', function (e) {
            e.stopPropagation(); // Prevent dropdown from closing
            const filterInputs = document.querySelectorAll('.search-filter');
            filterInputs.forEach(input => {
                if (input.type === 'checkbox') {
                    if (input.value === 'all') {
                        input.checked = true;
                    } else {
                        input.checked = false;
                        input.disabled = true;
                    }
                } else {
                    input.value = '';
                }
            });

            // Reset year picker selection
            if (yearPicker) {
                const event = new Event('click');
                if (resetYearBtn) resetYearBtn.dispatchEvent(event);
                updateYearLabel(); // Reset the year label
            }

            // Reset court checkboxes
            if (allCourtsCheckbox) {
                allCourtsCheckbox.checked = true;
                courtOptions.forEach(option => {
                    option.checked = false;
                    option.disabled = true;
                });
            }

            // Reset year button text if it exists
            const yearButton = document.querySelector('.dropdown-end label[tabindex="0"] span:contains("Year")');
            if (yearButton) {
                yearButton.textContent = 'Year';
            }
        });
    });

    // Initialize court filter functionality
    function initializeCourtFilter() {
        if (!allCourtsCheckbox) return;

        // Toggle "All Courts" checkbox
        allCourtsCheckbox.addEventListener('change', function () {
            courtOptions.forEach(option => {
                option.checked = false;
                option.disabled = this.checked;
            });
        });

        // Toggle "All Courts" when individual courts are selected
        courtOptions.forEach(option => {
            option.addEventListener('change', function () {
                const anyChecked = Array.from(courtOptions).some(opt => opt.checked);
                allCourtsCheckbox.checked = !anyChecked;
            });
        });
    }

    // Initialize year picker functionality
    function initializeYearPicker() {
        if (!yearPicker) return;

        let currentYear = new Date().getFullYear();
        let decadeStart = Math.floor(currentYear / 10) * 10;

        // Update decade display
        if (decadeDisplay) {
            decadeDisplay.textContent = `${decadeStart} - ${decadeStart + 19}`;
        }

        // Event listener for previous decade button
        if (prevDecadeBtn) {
            prevDecadeBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                decadeStart -= 10;
                if (decadeDisplay) {
                    decadeDisplay.textContent = `${decadeStart} - ${decadeStart + 19}`;
                }
                if (yearsList) {
                    populateYears(yearsList, decadeStart, currentYear);
                }
            });
        }

        // Event listener for next decade button
        if (nextDecadeBtn) {
            nextDecadeBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                decadeStart += 10;
                if (decadeDisplay) {
                    decadeDisplay.textContent = `${decadeStart} - ${decadeStart + 19}`;
                }
                if (yearsList) {
                    populateYears(yearsList, decadeStart, currentYear);
                }
            });
        }

        // Event listener for reset button
        if (resetYearBtn) {
            resetYearBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                if (yearFromInput) yearFromInput.value = '';
                if (yearToInput) yearToInput.value = '';
                if (yearsList) {
                    populateYears(yearsList, decadeStart, currentYear);
                }
                updateYearLabel();
            });
        }

        // Initial population of years
        if (yearsList) {
            populateYears(yearsList, decadeStart, currentYear);
        }
    }

    // Populate years in the dropdown
    function populateYears(container, decadeStart, currentYear) {
        container.innerHTML = '';

        const selectedFromYear = yearFromInput ? parseInt(yearFromInput.value) || null : null;
        const selectedToYear = yearToInput ? parseInt(yearToInput.value) || null : null;

        for (let year = decadeStart; year < decadeStart + 20; year++) {
            if (year <= currentYear) {
                const yearItem = document.createElement('div');
                yearItem.className = 'year-item';
                yearItem.textContent = year;

                // Highlight current year
                if (year === currentYear) {
                    yearItem.classList.add('current-year');
                }

                // Highlight selected years
                if (selectedFromYear !== null && selectedToYear !== null) {
                    if (year >= selectedFromYear && year <= selectedToYear) {
                        yearItem.classList.add('selected');
                    }
                } else if (selectedFromYear === year) {
                    yearItem.classList.add('selected');
                }

                // Add click event
                yearItem.addEventListener('click', function (e) {
                    e.stopPropagation();
                    handleYearItemClick(year, container, decadeStart, currentYear);
                });

                container.appendChild(yearItem);
            }
        }
    }

    // Handle year item click in new UI
    function handleYearItemClick(year, container, decadeStart, currentYear) {
        const selectedFromYear = yearFromInput ? parseInt(yearFromInput.value) || null : null;
        const selectedToYear = yearToInput ? parseInt(yearToInput.value) || null : null;

        if (selectedFromYear === null) {
            // First click - set from year
            if (yearFromInput) yearFromInput.value = year;
            if (yearButton) {
                const span = yearButton.querySelector('span');
                if (span) span.textContent = `${year}`;
            }
            populateYears(container, decadeStart, currentYear);
        } else if (selectedToYear === null) {
            // Second click - set to year
            if (year < selectedFromYear) {
                if (yearToInput) yearToInput.value = selectedFromYear;
                if (yearFromInput) yearFromInput.value = year;
                if (yearButton) {
                    const span = yearButton.querySelector('span');
                    if (span) span.textContent = `${year} - ${selectedFromYear}`;
                }
            } else {
                if (yearToInput) yearToInput.value = year;
                if (yearButton) {
                    const span = yearButton.querySelector('span');
                    if (span) span.textContent = `${selectedFromYear} - ${year}`;
                }
            }
            populateYears(container, decadeStart, currentYear);
        } else {
            // Reset and start new selection
            if (yearFromInput) yearFromInput.value = year;
            if (yearToInput) yearToInput.value = '';
            if (yearButton) {
                const span = yearButton.querySelector('span');
                if (span) span.textContent = `${year}`;
            }
            populateYears(container, decadeStart, currentYear);
        }
    }

    // Update the year label in the dropdown button
    function updateYearLabel(fromYear, toYear) {
        const yearLabel = document.querySelector('.dropdown-end label[tabindex="0"] span');
        if (!yearLabel) return;

        if (toYear) {
            // Year range
            yearLabel.innerHTML = `<i>${fromYear} - ${toYear}</i>`;
        } else if (fromYear) {
            // Single year
            yearLabel.innerHTML = `<i>${fromYear}</i>`;
        } else {
            // Reset to default
            yearLabel.innerHTML = 'Year';
        }
    }

    // Event listener for search button
    if (searchButton) {
        searchButton.addEventListener('click', function () {
            triggerSearch();
        });
    }

    // Search on Enter key
    if (searchInput) {
        searchInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                triggerSearch();
            }
        });
    }

    // Function to trigger search with all filters
    function triggerSearch() {
        if (!searchInput) return;

        const searchQuery = searchInput.value;
        if (searchQuery.length > 2) {
            // Use HTMX to trigger the search
            if (window.htmx) {
                htmx.trigger('#search-input', 'search');
            } else {
                // Fallback if htmx is not available
                const form = searchInput.closest('form');
                if (form) form.submit();
            }
        } else {
            alert('Please enter at least 3 characters to search');
        }
    }
});