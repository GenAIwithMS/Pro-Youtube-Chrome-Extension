// Content script for YouTube integration
console.log('YouTube RAG Chat extension loaded');

// Extract video ID from current page
function getCurrentVideoId() {
  const url = new URL(window.location.href);
  return url.searchParams.get('v');
}

// Listen for URL changes (YouTube is a SPA)
let currentVideoId = getCurrentVideoId();

// Observer for URL changes
const observer = new MutationObserver(() => {
  const newVideoId = getCurrentVideoId();
  if (newVideoId && newVideoId !== currentVideoId) {
    currentVideoId = newVideoId;
    console.log('Video changed to:', currentVideoId);
    
    // Notify background script about video change
    chrome.runtime.sendMessage({
      type: 'VIDEO_CHANGED',
      videoId: currentVideoId
    });
  }
});

// Start observing
observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Initial video detection
if (currentVideoId) {
  chrome.runtime.sendMessage({
    type: 'VIDEO_DETECTED',
    videoId: currentVideoId
  });
}


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_VIDEO_ID') {
    sendResponse({ videoId: getCurrentVideoId() });
  }
});

