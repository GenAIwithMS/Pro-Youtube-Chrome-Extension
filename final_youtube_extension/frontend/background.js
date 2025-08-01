// Background service worker for YouTube RAG Chat extension

// Handle extension installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('YouTube RAG Chat extension installed');
  
  // Set up initial storage
  chrome.storage.local.set({
    extensionInstalled: true,
    installDate: new Date().toISOString()
  });
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Background received message:', message);
  
  switch (message.type) {
    case 'VIDEO_DETECTED':
    case 'VIDEO_CHANGED':
      // Store current video ID
      chrome.storage.local.set({
        currentVideoId: message.videoId,
        lastUpdated: Date.now()
      });
      
      // Update badge to show video is detected
      chrome.action.setBadgeText({
        text: '●',
        tabId: sender.tab?.id
      });
      chrome.action.setBadgeBackgroundColor({
        color: '#4a9eff'
      });
      break;
      
    case 'GET_CURRENT_VIDEO':
      // Return current video ID
      chrome.storage.local.get(['currentVideoId'], (result) => {
        sendResponse({ videoId: result.currentVideoId });
      });
      return true; // Keep message channel open for async response
      
    case 'CLEAR_BADGE':
      chrome.action.setBadgeText({
        text: '',
        tabId: sender.tab?.id
      });
      break;
  }
});

// Handle tab updates to detect YouTube navigation
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    if (tab.url.includes('youtube.com/watch')) {
      const url = new URL(tab.url);
      const videoId = url.searchParams.get('v');
      
      if (videoId) {
        chrome.storage.local.set({
          currentVideoId: videoId,
          lastUpdated: Date.now()
        });
        
        // Update badge
        chrome.action.setBadgeText({
          text: '●',
          tabId: tabId
        });
        chrome.action.setBadgeBackgroundColor({
          color: '#4a9eff'
        });
      }
    } else {
      // Clear badge for non-YouTube pages
      chrome.action.setBadgeText({
        text: '',
        tabId: tabId
      });
    }
  }
});

// Handle tab activation (switching between tabs)
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  const tab = await chrome.tabs.get(activeInfo.tabId);
  
  if (tab.url && tab.url.includes('youtube.com/watch')) {
    const url = new URL(tab.url);
    const videoId = url.searchParams.get('v');
    
    if (videoId) {
      chrome.storage.local.set({
        currentVideoId: videoId,
        lastUpdated: Date.now()
      });
      
      chrome.action.setBadgeText({
        text: '●',
        tabId: activeInfo.tabId
      });
    }
  } else {
    chrome.action.setBadgeText({
      text: '',
      tabId: activeInfo.tabId
    });
  }
});

// Keep service worker alive
chrome.runtime.onStartup.addListener(() => {
  console.log('YouTube RAG Chat extension started');
});

// Handle extension icon click (optional - for debugging)
chrome.action.onClicked.addListener((tab) => {
  console.log('Extension icon clicked on tab:', tab.url);
});

