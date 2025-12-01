// Forbes Marshall SpotCheck - Enhanced JavaScript Functionality
class ParkingSystemEnhanced {
    constructor() {
        this.websocket = null;
        this.chartInstances = {};
        this.isRealTimeEnabled = true;
        this.notifications = [];
        this.init();
    }

    init() {
        this.initializeWebSocket();
        this.setupEventListeners();
        this.initializeCharts();
        this.loadNotifications();
        this.startPeriodicUpdates();
        this.setupKeyboardShortcuts();
        console.log('Forbes Marshall SpotCheck Enhanced System Initialized');
    }

    // WebSocket functionality
    initializeWebSocket() {
        if (typeof WebSocket !== 'undefined') {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/parking/`;
            
            try {
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    console.log('WebSocket connection established');
                    this.showNotification('Real-time updates connected', 'success');
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.websocket.onclose = () => {
                    console.log('WebSocket connection closed');
                    this.showNotification('Real-time connection lost', 'warning');
                    // Reconnect after 5 seconds
                    setTimeout(() => this.initializeWebSocket(), 5000);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.showNotification('Connection error', 'error');
                };
            } catch (error) {
                console.warn('WebSocket not available, falling back to polling');
                this.startPolling();
            }
        } else {
            this.startPolling();
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'slot_update':
                this.updateSlotDisplay(data.slot_id, data.status);
                break;
            case 'occupancy_change':
                this.updateOccupancyStats(data.stats);
                break;
            case 'new_vehicle':
                this.handleNewVehicle(data.vehicle);
                break;
            case 'system_alert':
                this.showNotification(data.message, data.level);
                break;
        }
    }

    // Notification system
    showNotification(message, type = 'info', duration = 5000) {
        const notification = {
            id: Date.now(),
            message,
            type,
            timestamp: new Date()
        };
        
        this.notifications.unshift(notification);
        this.displayNotification(notification);
        
        if (duration > 0) {
            setTimeout(() => this.removeNotification(notification.id), duration);
        }
    }

    displayNotification(notification) {
        const container = this.getNotificationContainer();
        const element = document.createElement('div');
        element.className = `notification notification-${notification.type} animate-fade-in`;
        element.id = `notification-${notification.id}`;
        
        element.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    ${this.getNotificationIcon(notification.type)}
                </div>
                <div class="notification-text">
                    <div class="notification-message">${notification.message}</div>
                    <div class="notification-time">${this.formatTime(notification.timestamp)}</div>
                </div>
                <button class="notification-close" onclick="parkingSystem.removeNotification(${notification.id})">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        container.appendChild(element);
    }

    removeNotification(id) {
        const element = document.getElementById(`notification-${id}`);
        if (element) {
            element.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => element.remove(), 300);
        }
        this.notifications = this.notifications.filter(n => n.id !== id);
    }

    getNotificationContainer() {
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
        return container;
    }

    getNotificationIcon(type) {
        const icons = {
            success: '<i class="fas fa-check-circle text-success"></i>',
            error: '<i class="fas fa-exclamation-circle text-danger"></i>',
            warning: '<i class="fas fa-exclamation-triangle text-warning"></i>',
            info: '<i class="fas fa-info-circle text-info"></i>'
        };
        return icons[type] || icons.info;
    }

    // Chart functionality
    initializeCharts() {
        this.initOccupancyChart();
        this.initTrendChart();
        this.initVehicleTypeChart();
    }

    initOccupancyChart() {
        const ctx = document.getElementById('occupancyChart');
        if (ctx && typeof Chart !== 'undefined') {
            this.chartInstances.occupancy = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Available', 'Occupied', 'Reserved'],
                    datasets: [{
                        data: [0, 0, 0],
                        backgroundColor: [
                            '#28a745',
                            '#dc3545',
                            '#ffc107'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }

    initTrendChart() {
        const ctx = document.getElementById('trendChart');
        if (ctx && typeof Chart !== 'undefined') {
            this.chartInstances.trend = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Occupancy Rate',
                        data: [],
                        borderColor: '#003366',
                        backgroundColor: 'rgba(0, 51, 102, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    }

    initVehicleTypeChart() {
        const ctx = document.getElementById('vehicleTypeChart');
        if (ctx && typeof Chart !== 'undefined') {
            this.chartInstances.vehicleType = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Sedan', 'SUV', 'Hatchback', 'Large'],
                    datasets: [{
                        label: 'Vehicle Count',
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            '#003366',
                            '#ff6600',
                            '#28a745',
                            '#ffc107'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
    }

    // Data update functions
    updateOccupancyStats(stats) {
        // Update chart
        if (this.chartInstances.occupancy) {
            this.chartInstances.occupancy.data.datasets[0].data = [
                stats.available || 0,
                stats.occupied || 0,
                stats.reserved || 0
            ];
            this.chartInstances.occupancy.update();
        }

        // Update dashboard cards
        this.updateStatsCards(stats);
    }

    updateStatsCards(stats) {
        const elements = {
            totalSlots: document.getElementById('total-slots'),
            availableSlots: document.getElementById('available-slots'),
            occupiedSlots: document.getElementById('occupied-slots'),
            occupancyRate: document.getElementById('occupancy-rate')
        };

        const total = (stats.available || 0) + (stats.occupied || 0) + (stats.reserved || 0);
        const occupancyPercent = total > 0 ? Math.round(((stats.occupied || 0) / total) * 100) : 0;

        if (elements.totalSlots) elements.totalSlots.textContent = total;
        if (elements.availableSlots) elements.availableSlots.textContent = stats.available || 0;
        if (elements.occupiedSlots) elements.occupiedSlots.textContent = stats.occupied || 0;
        if (elements.occupancyRate) elements.occupancyRate.textContent = `${occupancyPercent}%`;
    }

    updateSlotDisplay(slotId, status) {
        const slotElement = document.querySelector(`[data-slot-id="${slotId}"]`);
        if (slotElement) {
            // Remove existing status classes
            slotElement.classList.remove('available', 'occupied', 'reserved');
            // Add new status class
            slotElement.classList.add(status);
            
            // Add animation
            slotElement.classList.add('animate-pulse');
            setTimeout(() => slotElement.classList.remove('animate-pulse'), 1000);
        }
    }

    // Periodic updates (fallback for no WebSocket)
    startPolling() {
        setInterval(() => {
            if (this.isRealTimeEnabled) {
                this.fetchLatestData();
            }
        }, 30000); // Poll every 30 seconds
    }

    startPeriodicUpdates() {
        // Update time displays every minute
        setInterval(() => {
            this.updateTimeDisplays();
        }, 60000);

        // Update charts every 5 minutes
        setInterval(() => {
            this.updateCharts();
        }, 300000);
    }

    // API calls
    async fetchLatestData() {
        try {
            const response = await fetch('/api/parking/status/');
            if (response.ok) {
                const data = await response.json();
                this.updateOccupancyStats(data.stats);
                this.updateSlotDisplays(data.slots);
            }
        } catch (error) {
            console.error('Error fetching latest data:', error);
        }
    }

    async assignSlot(vehicleType, licensePlate) {
        try {
            const response = await fetch('/api/parking/assign/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    vehicle_type: vehicleType,
                    license_plate: licensePlate
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.showNotification(`Slot ${data.slot_id} assigned successfully`, 'success');
                return data;
            } else {
                const error = await response.json();
                this.showNotification(error.message || 'Assignment failed', 'error');
                return null;
            }
        } catch (error) {
            console.error('Error assigning slot:', error);
            this.showNotification('Network error during assignment', 'error');
            return null;
        }
    }

    async releaseSlot(slotId) {
        try {
            const response = await fetch(`/api/parking/release/${slotId}/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': this.getCSRFToken()
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.showNotification(`Slot ${slotId} released successfully`, 'success');
                return data;
            } else {
                const error = await response.json();
                this.showNotification(error.message || 'Release failed', 'error');
                return null;
            }
        } catch (error) {
            console.error('Error releasing slot:', error);
            this.showNotification('Network error during release', 'error');
            return null;
        }
    }

    // Utility functions
    getCSRFToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
    }

    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    updateTimeDisplays() {
        const elements = document.querySelectorAll('[data-timestamp]');
        elements.forEach(element => {
            const timestamp = new Date(element.dataset.timestamp);
            element.textContent = this.formatTime(timestamp);
        });
    }

    // Event listeners
    setupEventListeners() {
        // Slot click handlers
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('parking-slot-enhanced')) {
                this.handleSlotClick(e.target);
            }
        });

        // Form submissions
        const assignForm = document.getElementById('assign-slot-form');
        if (assignForm) {
            assignForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleAssignmentForm(e.target);
            });
        }

        // Real-time toggle
        const realtimeToggle = document.getElementById('realtime-toggle');
        if (realtimeToggle) {
            realtimeToggle.addEventListener('change', (e) => {
                this.isRealTimeEnabled = e.target.checked;
                this.showNotification(
                    `Real-time updates ${this.isRealTimeEnabled ? 'enabled' : 'disabled'}`,
                    'info'
                );
            });
        }
    }

    handleSlotClick(slotElement) {
        const slotId = slotElement.dataset.slotId;
        const status = slotElement.classList.contains('occupied') ? 'occupied' : 
                     slotElement.classList.contains('reserved') ? 'reserved' : 'available';
        
        // Show slot details modal or context menu
        this.showSlotDetails(slotId, status);
    }

    showSlotDetails(slotId, status) {
        // Implementation for slot details modal
        console.log(`Showing details for slot ${slotId} (${status})`);
    }

    // Keyboard shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 'n':
                        e.preventDefault();
                        this.openAssignmentModal();
                        break;
                    case 'h':
                        e.preventDefault();
                        this.showHelp();
                        break;
                }
            }
        });
    }

    refreshData() {
        this.fetchLatestData();
        this.showNotification('Data refreshed', 'info');
    }

    openAssignmentModal() {
        const modal = document.getElementById('assignment-modal');
        if (modal) {
            // Open modal logic
            console.log('Opening assignment modal');
        }
    }

    showHelp() {
        const helpModal = document.getElementById('help-modal');
        if (helpModal) {
            // Show help modal
            console.log('Showing help');
        }
    }

    // Export functionality
    exportData(format = 'csv') {
        const endpoint = `/api/parking/export/?format=${format}`;
        window.open(endpoint, '_blank');
        this.showNotification(`Exporting data as ${format.toUpperCase()}`, 'info');
    }

    // Print functionality
    printReport() {
        window.print();
    }

    // Clean up
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        Object.values(this.chartInstances).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        console.log('Forbes Marshall SpotCheck Enhanced System Destroyed');
    }
}

// Initialize the enhanced parking system
let parkingSystem;
document.addEventListener('DOMContentLoaded', () => {
    parkingSystem = new ParkingSystemEnhanced();
});

// Global utility functions
function assignSlot() {
    const vehicleType = document.getElementById('vehicle-type')?.value;
    const licensePlate = document.getElementById('license-plate')?.value;
    
    if (vehicleType && licensePlate) {
        parkingSystem.assignSlot(vehicleType, licensePlate);
    } else {
        parkingSystem.showNotification('Please fill in all fields', 'warning');
    }
}

function releaseSlot(slotId) {
    if (confirm(`Are you sure you want to release slot ${slotId}?`)) {
        parkingSystem.releaseSlot(slotId);
    }
}

function toggleRealTime() {
    parkingSystem.isRealTimeEnabled = !parkingSystem.isRealTimeEnabled;
    parkingSystem.showNotification(
        `Real-time updates ${parkingSystem.isRealTimeEnabled ? 'enabled' : 'disabled'}`,
        'info'
    );
}

function exportReport(format) {
    parkingSystem.exportData(format);
}

function printReport() {
    parkingSystem.printReport();
}

// Service Worker registration for offline capability
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}