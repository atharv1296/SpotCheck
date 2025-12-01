// Forbes Marshall SpotCheck - Service Worker for Offline Functionality
const CACHE_NAME = 'forbes-marshall-spotcheck-v1.0.0';
const urlsToCache = [
    '/',
    '/static/css/dashboard.css',
    '/static/css/enhancements.css',
    '/static/js/dashboard.js',
    '/static/js/enhanced.js',
    '/static/images/forbes-marshall-logo.png',
    '/static/images/favicon.ico',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://cdn.jsdelivr.net/npm/chart.js'
];

// Install event - cache resources
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
});

// Fetch event - serve cached content when offline
self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // Return cached version or fetch from network
                if (response) {
                    return response;
                }
                
                return fetch(event.request).then(response => {
                    // Check if we received a valid response
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response;
                    }
                    
                    // Clone the response
                    const responseToCache = response.clone();
                    
                    caches.open(CACHE_NAME)
                        .then(cache => {
                            cache.put(event.request, responseToCache);
                        });
                        
                    return response;
                });
            })
            .catch(() => {
                // Return offline page for navigation requests
                if (event.request.destination === 'document') {
                    return caches.match('/offline.html');
                }
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Background sync for offline actions
self.addEventListener('sync', event => {
    if (event.tag === 'background-sync') {
        event.waitUntil(doBackgroundSync());
    }
});

async function doBackgroundSync() {
    // Sync any queued actions when online
    const actions = await getQueuedActions();
    
    for (const action of actions) {
        try {
            await processAction(action);
            await removeFromQueue(action.id);
        } catch (error) {
            console.error('Failed to sync action:', error);
        }
    }
}

async function getQueuedActions() {
    const cache = await caches.open('actions-queue');
    const requests = await cache.keys();
    return requests.map(req => req.url);
}

async function processAction(action) {
    // Process queued actions
    return fetch(action.url, action.options);
}

async function removeFromQueue(actionId) {
    const cache = await caches.open('actions-queue');
    await cache.delete(actionId);
}

// Push notification handling
self.addEventListener('push', event => {
    const options = {
        body: event.data ? event.data.text() : 'New parking update available',
        icon: '/static/images/favicon.ico',
        badge: '/static/images/forbes-marshall-logo.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Dashboard',
                icon: '/static/images/favicon.ico'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/static/images/favicon.ico'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('Forbes Marshall SpotCheck', options)
    );
});

// Notification click handling
self.addEventListener('notificationclick', event => {
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Message handling from main thread
self.addEventListener('message', event => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});