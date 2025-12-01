// Forbes Marshall Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    initializeAnimations();
    initializeSidebar();
});

function initializeDashboard() {
    // Initialize parking data
    updateParkingStats();
    
    // Setup real-time updates
    setInterval(updateParkingStats, 30000); // Update every 30 seconds
    
    // Initialize vehicle type selector
    initializeVehicleTypeSelector();
    
    // Initialize parking spot interactions
    initializeParkingSpots();
}

function initializeAnimations() {
    // Add entrance animations to cards
    const cards = document.querySelectorAll('.card, .stats-card, .info-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    // Add hover effects
    addHoverEffects();
    
    // Initialize floating animations
    initializeFloatingAnimations();
}

function initializeSidebar() {
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebar-overlay');
    const sidebarClose = document.getElementById('sidebar-close');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('show');
            sidebarOverlay.classList.toggle('show');
        });
    }
    
    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', function() {
            sidebar.classList.remove('show');
            this.classList.remove('show');
        });
    }
    
    if (sidebarClose) {
        sidebarClose.addEventListener('click', function() {
            sidebar.classList.remove('show');
            sidebarOverlay.classList.remove('show');
        });
    }
}

function initializeVehicleTypeSelector() {
    const vehicleTypes = document.querySelectorAll('.vehicle-type');
    
    vehicleTypes.forEach(type => {
        type.addEventListener('click', function() {
            // Remove active class from all types
            vehicleTypes.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked type
            this.classList.add('active');
            
            // Get vehicle type
            const vehicleType = this.dataset.vehicle;
            
            // Show corresponding parking section
            showParkingSection(vehicleType);
        });
    });
}

function showParkingSection(vehicleType) {
    // Remove active class from all vehicle type buttons
    const allButtons = document.querySelectorAll('.type-button');
    allButtons.forEach(button => {
        button.classList.remove('active');
    });
    
    // Add active class to clicked button
    const activeButton = document.querySelector(`[data-vehicle="${vehicleType}"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
    
    // Hide all parking sections
    const allSections = document.querySelectorAll('.parking-section');
    allSections.forEach(section => {
        section.style.display = 'none';
    });
    
    // Show selected section
    const selectedSection = document.getElementById(vehicleType + '-section');
    if (selectedSection) {
        selectedSection.style.display = 'block';
        selectedSection.style.opacity = '0';
        selectedSection.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            selectedSection.style.transition = 'all 0.6s ease';
            selectedSection.style.opacity = '1';
            selectedSection.style.transform = 'translateY(0)';
        }, 100);
    }
    
    // Update stats for selected vehicle type
    updateVehicleTypeStats(vehicleType);
}

function updateVehicleTypeStats(vehicleType) {
    const stats = {
        hatchback: { total: 25, occupied: 15, available: 10 },
        sedan: { total: 20, occupied: 12, available: 8 },
        suv: { total: 15, occupied: 8, available: 7 },
        large: { total: 10, occupied: 6, available: 4 }
    };
    
    const vehicleStats = stats[vehicleType] || stats.hatchback;
    
    // Update display
    const totalElement = document.getElementById('total-vehicles');
    const availableElement = document.getElementById('available-spots');
    
    if (totalElement) {
        animateNumber(totalElement, vehicleStats.occupied);
    }
    
    if (availableElement) {
        animateNumber(availableElement, vehicleStats.available);
    }
}

function initializeParkingSpots() {
    const parkingSpots = document.querySelectorAll('.parking-spot, .parking-spot-vertical');
    
    parkingSpots.forEach(spot => {
        spot.addEventListener('click', function() {
            const slotId = this.dataset.slot;
            const isOccupied = this.classList.contains('spot-occupied');
            
            if (isOccupied) {
                showVehicleDetails(slotId);
            } else {
                showEmptySlotOptions(slotId);
            }
        });
        
        // Add hover effects
        spot.addEventListener('mouseenter', function() {
            if (!this.classList.contains('spot-occupied')) {
                this.style.borderColor = '#7ED321';
                this.style.boxShadow = '0 0 20px rgba(126, 211, 33, 0.3)';
            }
        });
        
        spot.addEventListener('mouseleave', function() {
            if (!this.classList.contains('spot-occupied')) {
                this.style.borderColor = '#4a5568';
                this.style.boxShadow = 'inset 0 2px 4px rgba(0, 0, 0, 0.1)';
            }
        });
    });
}

function showVehicleDetails(slotId) {
    // Sample vehicle data - in real app, this would come from API
    const vehicleData = {
        'H1': { number: 'MH12AB1234', type: 'Hatchback', owner: 'John Doe', entry: '09:30 AM' },
        'H3': { number: 'GJ05CD5678', type: 'Hatchback', owner: 'Jane Smith', entry: '10:45 AM' },
        'S1': { number: 'KA03GH3456', type: 'Sedan', owner: 'Mike Johnson', entry: '11:20 AM' },
        'U1': { number: 'RJ14KL1234', type: 'SUV', owner: 'Sarah Wilson', entry: '08:15 AM' },
        'L1': { number: 'PB03OP9012', type: 'Large Vehicle', owner: 'David Brown', entry: '07:45 AM' }
    };
    
    const vehicle = vehicleData[slotId];
    if (vehicle) {
        // Show modal with vehicle details
        console.log(`Vehicle Details for Slot ${slotId}:`, vehicle);
        // In real implementation, you would populate and show a modal here
    }
}

function showEmptySlotOptions(slotId) {
    console.log(`Empty slot ${slotId} clicked - showing options`);
    // In real implementation, you would show options to assign a vehicle
}

function updateParkingStats() {
    // Simulate real-time data updates
    const totalVehicles = document.getElementById('total-vehicles');
    const availableSpots = document.getElementById('available-spots');
    
    // Generate random updates within realistic ranges
    const currentTotal = parseInt(totalVehicles?.textContent || '420');
    const variation = Math.floor(Math.random() * 10) - 5; // ±5 vehicles
    const newTotal = Math.max(0, Math.min(500, currentTotal + variation));
    const newAvailable = 500 - newTotal;
    
    if (totalVehicles) {
        animateNumber(totalVehicles, newTotal);
    }
    
    if (availableSpots) {
        animateNumber(availableSpots, newAvailable);
    }
    
    // Update occupancy percentage
    const occupancyElements = document.querySelectorAll('.level-stat-number');
    if (occupancyElements.length >= 3) {
        const occupancyPercent = Math.round((newTotal / 500) * 100);
        animateNumber(occupancyElements[2], occupancyPercent, '%');
    }
}

function animateNumber(element, targetValue, suffix = '') {
    const startValue = parseInt(element.textContent) || 0;
    const duration = 1000;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const currentValue = Math.round(startValue + (targetValue - startValue) * easeOutCubic);
        
        element.textContent = currentValue + suffix;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function addHoverEffects() {
    // Add magnetic hover effects to cards
    const cards = document.querySelectorAll('.card, .stats-card, .info-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Add button hover effects
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

function initializeFloatingAnimations() {
    // Floating animation for FAB
    const fab = document.querySelector('.fab');
    if (fab) {
        let floatDirection = 1;
        setInterval(() => {
            fab.style.transform = `translateY(${floatDirection * 3}px)`;
            floatDirection *= -1;
        }, 2000);
    }
    
    // Subtle floating for stats numbers
    const statsNumbers = document.querySelectorAll('.level-stat-number');
    statsNumbers.forEach((stat, index) => {
        setTimeout(() => {
            let direction = 1;
            setInterval(() => {
                stat.style.transform = `translateY(${direction * 1}px)`;
                direction *= -1;
            }, 3000 + (index * 500));
        }, index * 200);
    });
}

// Search functionality
function initializeSearch() {
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            handleSearch(query);
        });
    }
}

function handleSearch(query) {
    // Implement search functionality
    console.log('Searching for:', query);
    // In real implementation, you would filter parking spots, vehicles, etc.
}

// Real-time activity feed
function updateActivityFeed() {
    const activities = [
        { type: 'entry', vehicle: 'MH12AB1234', slot: '105', time: '2 minutes ago' },
        { type: 'exit', vehicle: 'GJ01CD5678', slot: '208', time: '5 minutes ago' },
        { type: 'entry', vehicle: 'DL03EF9012', slot: '104', time: '8 minutes ago' }
    ];
    
    // In real implementation, you would update the activity feed with fresh data
    console.log('Activity feed updated:', activities);
}

// Quick actions
function showQuickScan() {
    console.log('Quick scan initiated');
    // In real implementation, you would show QR code scanner or similar
}

// Initialize search and activity feed updates
document.addEventListener('DOMContentLoaded', function() {
    initializeSearch();
    
    // Update activity feed every minute
    setInterval(updateActivityFeed, 60000);
});

// Handle quick actions modal
function initializeQuickActions() {
    const quickActionBtns = document.querySelectorAll('[data-bs-toggle="modal"]');
    quickActionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const target = this.dataset.bsTarget;
            console.log('Opening modal:', target);
        });
    });
}

// Handle slot clicks
function handleSlotClick(slotId, isOccupied) {
    const slot = document.querySelector(`[data-slot="${slotId}"]`);
    if (!slot) return;
    
    if (isOccupied) {
        // Show vehicle details modal
        showVehicleDetailsModal(slotId);
    } else {
        // Show assign vehicle modal
        showAssignVehicleModal(slotId);
    }
    
    // Add visual feedback
    slot.style.transform = 'scale(0.95)';
    setTimeout(() => {
        slot.style.transform = 'scale(1)';
    }, 150);
}

function showVehicleDetailsModal(slotId) {
    // Sample vehicle data - in real app, this would come from API
    const vehicleData = {
        'H1': { number: 'MH12AB1234', type: 'Hatchback', owner: 'John Doe', entry: '09:30 AM', duration: '2h 30m' },
        'H3': { number: 'GJ05CD5678', type: 'Hatchback', owner: 'Jane Smith', entry: '10:45 AM', duration: '1h 15m' },
        'H5': { number: 'DL01EF2345', type: 'Hatchback', owner: 'Alex Johnson', entry: '11:20 AM', duration: '45m' },
        'S1': { number: 'KA03GH3456', type: 'Sedan', owner: 'Mike Johnson', entry: '08:15 AM', duration: '4h 45m' },
        'S3': { number: 'TN09IJ7890', type: 'Sedan', owner: 'Sarah Davis', entry: '09:30 AM', duration: '3h 30m' },
        'U2': { number: 'RJ14KL1234', type: 'SUV', owner: 'Sarah Wilson', entry: '07:45 AM', duration: '5h 15m' },
        'U4': { number: 'UP32MN5678', type: 'SUV', owner: 'David Brown', entry: '10:00 AM', duration: '2h' },
        'L1': { number: 'PB03OP9012', type: 'Large Vehicle', owner: 'Robert Lee', entry: '06:30 AM', duration: '6h 30m' },
        'L3': { number: 'HR05QR3456', type: 'Large Vehicle', owner: 'Lisa White', entry: '08:45 AM', duration: '4h 15m' }
    };
    
    const vehicle = vehicleData[slotId];
    if (vehicle) {
        const modalHtml = `
            <div class="modal fade" id="vehicleDetailsModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Vehicle Details - Slot ${slotId}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-6">
                                    <strong>Vehicle Number:</strong><br>
                                    <span class="text-primary">${vehicle.number}</span>
                                </div>
                                <div class="col-6">
                                    <strong>Vehicle Type:</strong><br>
                                    ${vehicle.type}
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-6">
                                    <strong>Owner:</strong><br>
                                    ${vehicle.owner}
                                </div>
                                <div class="col-6">
                                    <strong>Entry Time:</strong><br>
                                    ${vehicle.entry}
                                </div>
                            </div>
                            <hr>
                            <div class="row">
                                <div class="col-12">
                                    <strong>Duration:</strong><br>
                                    <span class="text-success">${vehicle.duration}</span>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-danger" onclick="releaseSlot('${slotId}')">Release Slot</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        const existingModal = document.getElementById('vehicleDetailsModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('vehicleDetailsModal'));
        modal.show();
    }
}

function showAssignVehicleModal(slotId) {
    const modalHtml = `
        <div class="modal fade" id="assignVehicleModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Assign Slot ${slotId}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="assignVehicleForm">
                            <div class="mb-3">
                                <label for="vehicleNumber" class="form-label">Vehicle Number</label>
                                <input type="text" class="form-control" id="vehicleNumber" placeholder="e.g., MH12AB1234" required>
                            </div>
                            <div class="mb-3">
                                <label for="ownerName" class="form-label">Owner Name</label>
                                <input type="text" class="form-control" id="ownerName" placeholder="e.g., John Doe" required>
                            </div>
                            <div class="mb-3">
                                <label for="vehicleType" class="form-label">Vehicle Type</label>
                                <select class="form-select" id="vehicleType" required>
                                    <option value="">Select Vehicle Type</option>
                                    <option value="Hatchback">Hatchback</option>
                                    <option value="Sedan">Sedan</option>
                                    <option value="SUV">SUV</option>
                                    <option value="Large Vehicle">Large Vehicle</option>
                                </select>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="button" class="btn btn-primary" onclick="assignVehicle('${slotId}')">Assign Slot</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('assignVehicleModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('assignVehicleModal'));
    modal.show();
}

function assignVehicle(slotId) {
    const form = document.getElementById('assignVehicleForm');
    const vehicleNumber = document.getElementById('vehicleNumber').value.trim();
    const ownerName = document.getElementById('ownerName').value.trim();
    const vehicleType = document.getElementById('vehicleType').value;
    
    if (!vehicleNumber || !ownerName || !vehicleType) {
        alert('Please fill all fields');
        return;
    }
    
    // Update slot status
    const slot = document.querySelector(`[data-slot="${slotId}"]`);
    if (slot) {
        slot.classList.remove('available');
        slot.classList.add('occupied');
        slot.querySelector('.slot-icon').textContent = '🚗';
        slot.querySelector('.slot-status').textContent = 'Occupied';
        slot.setAttribute('onclick', `handleSlotClick('${slotId}', true)`);
    }
    
    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('assignVehicleModal'));
    modal.hide();
    
    // Show success message
    showNotification(`Vehicle ${vehicleNumber} assigned to slot ${slotId}`, 'success');
    
    // Update stats
    updateSectionStats();
}

function releaseSlot(slotId) {
    // Update slot status
    const slot = document.querySelector(`[data-slot="${slotId}"]`);
    if (slot) {
        slot.classList.remove('occupied');
        slot.classList.add('available');
        slot.querySelector('.slot-icon').textContent = '🅿️';
        slot.querySelector('.slot-status').textContent = 'Available';
        slot.setAttribute('onclick', `handleSlotClick('${slotId}', false)`);
    }
    
    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('vehicleDetailsModal'));
    modal.hide();
    
    // Show success message
    showNotification(`Slot ${slotId} has been released`, 'success');
    
    // Update stats
    updateSectionStats();
}

function showNotification(message, type = 'info') {
    const alertClass = type === 'success' ? 'alert-success' : type === 'error' ? 'alert-danger' : 'alert-info';
    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 9999; min-width: 300px;">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', alertHtml);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        const alert = document.querySelector('.alert');
        if (alert) {
            alert.remove();
        }
    }, 3000);
}

function updateSectionStats() {
    const sections = ['hatchback', 'sedan', 'suv', 'large'];
    
    sections.forEach(section => {
        const sectionElement = document.getElementById(section + '-section');
        if (sectionElement) {
            const allSlots = sectionElement.querySelectorAll('.parking-slot');
            const occupiedSlots = sectionElement.querySelectorAll('.parking-slot.occupied');
            const availableSlots = allSlots.length - occupiedSlots.length;
            
            // Update section stats
            const availableElement = document.getElementById(section + '-available');
            const occupiedElement = document.getElementById(section + '-occupied');
            
            if (availableElement) {
                availableElement.textContent = `${availableSlots} Available`;
            }
            
            if (occupiedElement) {
                occupiedElement.textContent = `${occupiedSlots.length} Occupied`;
            }
        }
    });
    
    // Update main stats
    updateMainStats();
}

function updateMainStats() {
    const allSlots = document.querySelectorAll('.parking-slot');
    const occupiedSlots = document.querySelectorAll('.parking-slot.occupied');
    const availableSlots = allSlots.length - occupiedSlots.length;
    
    const totalVehiclesElement = document.getElementById('total-vehicles');
    const availableSpotsElement = document.getElementById('available-spots');
    
    if (totalVehiclesElement) {
        totalVehiclesElement.textContent = occupiedSlots.length;
    }
    
    if (availableSpotsElement) {
        availableSpotsElement.textContent = availableSlots;
    }
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('Dashboard error:', event.error);
});

// Performance monitoring
function monitorPerformance() {
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
            }, 0);
        });
    }
}

monitorPerformance();
