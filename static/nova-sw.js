const CACHE = 'nova-learn-v1';
const ASSETS = [
  '/nova',
  '/workspace/nova_curriculum.json',
  '/workspace/nova_progress.json',
  '/workspace/nova_vignettes.json',
  '/static/nova-manifest.json'
];

self.addEventListener('install', function(e) {
  e.waitUntil(
    caches.open(CACHE).then(function(cache) {
      return cache.addAll(ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function(e) {
  e.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(keys.filter(function(k){ return k !== CACHE; }).map(function(k){ return caches.delete(k); }));
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function(e) {
  // Network first for API calls, cache first for static assets
  var url = e.request.url;
  if (url.includes('/learn/chat') || url.includes('/api/nova/progress')) {
    e.respondWith(
      fetch(e.request).catch(function() {
        return new Response(JSON.stringify({response: 'Nova is offline. Check your connection.'}),
          {headers: {'Content-Type': 'application/json'}});
      })
    );
  } else {
    e.respondWith(
      caches.match(e.request).then(function(cached) {
        return cached || fetch(e.request).then(function(response) {
          var clone = response.clone();
          caches.open(CACHE).then(function(cache){ cache.put(e.request, clone); });
          return response;
        });
      }).catch(function() {
        return caches.match('/nova');
      })
    );
  }
});
