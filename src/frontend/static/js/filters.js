document.addEventListener('DOMContentLoaded', function () {
    // Search input and button
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchForm = document.getElementById('search-form');

    // Reset buttons
    const resetButtons = document.querySelectorAll('.reset-filters');

    // Year picker elements
    const yearPicker = document.querySelector('.year-picker');
    const decadeDisplay = document.querySelector('.decade-display');
    const prevDecadeBtn = document.querySelector('.prev-decade-btn');
    const nextDecadeBtn = document.querySelector('.next-decade-btn');
    const yearsList = document.querySelector('.years-list');
    const resetYearBtn = document.querySelector('.reset-year-filters');
    const yearFromSelect = document.querySelector('select[name="year_from"]');
    const yearToSelect = document.querySelector('select[name="year_to"]');
    const yearButton = document.querySelector('.dropdown-end button[tabindex="0"]');

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
                if (yearFromSelect) yearFromSelect.value = '';
                if (yearToSelect) yearToSelect.value = '';
                updateYearLabel();
                const currentYear = new Date().getFullYear();
                const decadeStart = Math.floor(currentYear / 10) * 10;
                populateYears(yearsList, decadeStart, currentYear);
            }

            // Reset court checkboxes
            if (allCourtsCheckbox) {
                allCourtsCheckbox.checked = true;
                courtOptions.forEach(option => {
                    option.checked = false;
                    option.disabled = true;
                });
            }

            // Trigger search if there's a query
            if (searchForm && searchInput.value.trim().length >= 3) {
                htmx.trigger(searchForm, 'submit');
            }
        });
    });

    // Event listener for year reset button
    if (resetYearBtn) {
        resetYearBtn.addEventListener('click', function (e) {
            e.stopPropagation(); // Prevent dropdown from closing

            // Reset year selects
            if (yearFromSelect) yearFromSelect.value = '';
            if (yearToSelect) yearToSelect.value = '';

            // Reset year button label
            if (yearButton) {
                const span = yearButton.querySelector('span:last-child');
                if (span) span.textContent = 'Year';
            }

            // Trigger search if there's a query
            if (searchForm && searchInput.value.trim().length >= 3) {
                htmx.trigger(searchForm, 'submit');
            }
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

        // Initial population of years
        if (yearsList) {
            populateYears(yearsList, decadeStart, currentYear);
        }

        // Keep dropdown open when clicking inside
        yearPicker.addEventListener('click', function (e) {
            e.stopPropagation();
        });
    }

    // Populate years in the dropdown
    function populateYears(container, decadeStart, currentYear) {
        if (!container) return;
        container.innerHTML = '';

        const selectedFromYear = yearFromSelect ? parseInt(yearFromSelect.value) || null : null;
        const selectedToYear = yearToSelect ? parseInt(yearToSelect.value) || null : null;

        for (let year = decadeStart; year < decadeStart + 20; year++) {
            if (year <= currentYear) {
                const yearItem = document.createElement('button');
                yearItem.type = 'button';
                yearItem.className = 'year-item btn btn-ghost btn-sm';
                yearItem.textContent = year;

                // Highlight current year
                if (year === currentYear) {
                    yearItem.classList.add('current-year');
                }

                // Highlight selected years
                if (selectedFromYear !== null && selectedToYear !== null) {
                    if (year >= Math.min(selectedFromYear, selectedToYear) &&
                        year <= Math.max(selectedFromYear, selectedToYear)) {
                        yearItem.classList.add('selected');
                        yearItem.classList.remove('btn-ghost');
                    }
                } else if (selectedFromYear === year) {
                    yearItem.classList.add('selected');
                    yearItem.classList.remove('btn-ghost');
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
        const selectedFromYear = yearFromSelect ? parseInt(yearFromSelect.value) || null : null;
        const selectedToYear = yearToSelect ? parseInt(yearToSelect.value) || null : null;

        if (selectedFromYear === null) {
            // First click - set from year
            if (yearFromSelect) yearFromSelect.value = year;
            updateYearLabel(year);
            populateYears(container, decadeStart, currentYear);
        } else if (selectedToYear === null) {
            // Second click - set to year
            if (year < selectedFromYear) {
                if (yearToSelect) yearToSelect.value = selectedFromYear;
                if (yearFromSelect) yearFromSelect.value = year;
                updateYearLabel(year, selectedFromYear);
            } else {
                if (yearToSelect) yearToSelect.value = year;
                updateYearLabel(selectedFromYear, year);
            }
            populateYears(container, decadeStart, currentYear);

            // Trigger search if there's a query
            if (searchForm && searchInput.value.trim().length >= 3) {
                htmx.trigger(searchForm, 'submit');
            }
        } else {
            // Reset and start new selection
            if (yearFromSelect) yearFromSelect.value = year;
            if (yearToSelect) yearToSelect.value = '';
            updateYearLabel(year);
            populateYears(container, decadeStart, currentYear);
        }
    }

    // Update the year label in the dropdown button
    function updateYearLabel(fromYear, toYear) {
        if (!yearButton) return;
        const span = yearButton.querySelector('span:last-child');
        if (!span) return;

        if (toYear) {
            // Year range
            span.textContent = `${Math.min(fromYear, toYear)} - ${Math.max(fromYear, toYear)}`;
        } else if (fromYear) {
            // Single year
            span.textContent = `${fromYear}`;
        } else {
            // Reset to default
            span.textContent = 'Year';
        }
    }

    // Initialize court filter functionality
    function initializeCourtFilter() {
        if (!allCourtsCheckbox) return;

        // Toggle "All Courts" checkbox
        allCourtsCheckbox.addEventListener('change', function () {
            courtOptions.forEach(option => {
                option.checked = false;
                option.disabled = this.checked;
            });

            // Trigger search if there's a query
            if (searchForm && searchInput.value.trim().length >= 3) {
                htmx.trigger(searchForm, 'submit');
            }
        });

        // Toggle "All Courts" when individual courts are selected
        courtOptions.forEach(option => {
            option.addEventListener('change', function () {
                const anyChecked = Array.from(courtOptions).some(opt => opt.checked);
                allCourtsCheckbox.checked = !anyChecked;

                // Trigger search if there's a query
                if (searchForm && searchInput.value.trim().length >= 3) {
                    htmx.trigger(searchForm, 'submit');
                }
            });
        });
    }

    // Add event listeners for select changes
    if (yearFromSelect) {
        yearFromSelect.addEventListener('change', function () {
            const fromYear = parseInt(this.value) || null;
            const toYear = yearToSelect ? parseInt(yearToSelect.value) || null : null;

            if (fromYear) {
                updateYearLabel(fromYear, toYear);

                // Trigger search if there's a query
                if (searchForm && searchInput.value.trim().length >= 3) {
                    htmx.trigger(searchForm, 'submit');
                }
            } else if (!toYear) {
                // Both are empty, reset label
                updateYearLabel();
            } else {
                // Only to year is set
                updateYearLabel(null, toYear);
            }
        });
    }

    if (yearToSelect) {
        yearToSelect.addEventListener('change', function () {
            const fromYear = yearFromSelect ? parseInt(yearFromSelect.value) || null : null;
            const toYear = parseInt(this.value) || null;

            if (toYear && fromYear) {
                updateYearLabel(fromYear, toYear);

                // Trigger search if there's a query
                if (searchForm && searchInput.value.trim().length >= 3) {
                    htmx.trigger(searchForm, 'submit');
                }
            } else if (toYear) {
                // Only to year is set
                updateYearLabel(null, toYear);
            } else if (!fromYear) {
                // Both are empty, reset label
                updateYearLabel();
            }
        });
    }
});