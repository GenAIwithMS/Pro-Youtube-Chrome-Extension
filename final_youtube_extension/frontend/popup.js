const API_BASE_URL = 'http://localhost:8000';

let currentVideoId = '';
let isProcessed = false;
let isProcessing = false;
let currentUserId = 'user_' + Math.random().toString(36).substr(2, 9);
let userSubscription = null;
let selectedPlan = 'pro';

// DOM elements
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const statusBar = document.getElementById('statusBar');
const loading = document.getElementById('loading');
const planBadge = document.getElementById('planBadge');
const planDetails = document.getElementById('planDetails');
const upgradeBtn = document.getElementById('upgradeBtn');
const usageStats = document.getElementById('usageStats');

// Premium feature elements
const brainModeBtn = document.getElementById('brainModeBtn');
const timelineBtn = document.getElementById('timelineBtn');
const voiceTab = document.getElementById('voiceTab');
const featuresTab = document.getElementById('featuresTab');
const graphTab = document.getElementById('graphTab');

// Modal elements
const upgradeModal = document.getElementById('upgradeModal');
const closeModal = document.getElementById('closeModal');
const startTrial = document.getElementById('startTrial');

// Tab system
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
  await initializeExtension();
  setupEventListeners();
  setupTabSystem();
  setupUpgradeModal();
  await loadUserSubscription();
});

async function initializeExtension() {
  try {
    // Get current tab URL to extract video ID
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (tab.url && tab.url.includes('youtube.com/watch')) {
      const url = new URL(tab.url);
      const videoId = url.searchParams.get('v');
      
      if (videoId) {
        currentVideoId = videoId;
        updateStatus(`Detected video: ${videoId}`, 'info');
        
        // Check if video is already processed
        await checkVideoStatus(videoId);
        
        // Load stored messages for this video
        await loadStoredMessages(videoId);
      } else {
        updateStatus('No video ID found in URL', 'error');
      }
    } else {
      updateStatus('Please navigate to a YouTube video', 'info');
    }
  } catch (error) {
    console.error('Error initializing extension:', error);
    updateStatus('Error initializing extension', 'error');
  }
}

async function loadUserSubscription() {
  try {
    const response = await fetch(`${API_BASE_URL}/subscription/${currentUserId}`);
    if (response.ok) {
      const data = await response.json();
      userSubscription = data.subscription;
      updateSubscriptionUI(userSubscription, data.stats);
    } else {
      // Create default free subscription
      userSubscription = {
        plan: 'free',
        status: 'active',
        features_used_today: {}
      };
      updateSubscriptionUI(userSubscription, {});
    }
  } catch (error) {
    console.error('Error loading subscription:', error);
    // Default to free plan
    userSubscription = {
      plan: 'free',
      status: 'active',
      features_used_today: {}
    };
    updateSubscriptionUI(userSubscription, {});
  }
}

function updateSubscriptionUI(subscription, stats) {
  // Update plan badge
  planBadge.className = `plan-badge plan-${subscription.plan.replace('_', '-')}`;
  planBadge.textContent = subscription.plan.charAt(0).toUpperCase() + subscription.plan.slice(1).replace('_', '+');
  
  // Update plan details
  if (subscription.plan === 'free') {
    const chatUsage = subscription.features_used_today?.basic_chat || 0;
    planDetails.textContent = `${50 - chatUsage} chats left today`;
    usageStats.textContent = `Chats used today: ${chatUsage}/50`;
    
    if (chatUsage >= 50) {
      usageStats.style.color = '#ff4757';
      usageStats.textContent += ' - Limit reached!';
    }
  } else {
    planDetails.textContent = 'Unlimited chats';
    usageStats.textContent = `Premium features active`;
    usageStats.style.color = '#4a9eff';
  }
  
  // Update premium feature availability
  updatePremiumFeatureAccess(subscription.plan);
  
  // Show/hide upgrade button
  if (subscription.plan === 'free') {
    upgradeBtn.style.display = 'block';
  } else {
    upgradeBtn.style.display = 'none';
  }
}

function updatePremiumFeatureAccess(plan) {
  const planHierarchy = ['free', 'pro', 'power_plus', 'creator'];
  const currentPlanLevel = planHierarchy.indexOf(plan);
  
  // Brain Mode and Timeline (Pro+)
  if (currentPlanLevel >= 1) {
    brainModeBtn.classList.remove('locked');
    timelineBtn.classList.remove('locked');
    voiceTab.classList.remove('locked');
  } else {
    brainModeBtn.classList.add('locked');
    timelineBtn.classList.add('locked');
    voiceTab.classList.add('locked');
  }
  
  // Advanced features (Power+)
  if (currentPlanLevel >= 2) {
    featuresTab.classList.remove('locked');
    graphTab.classList.remove('locked');
  } else {
    featuresTab.classList.add('locked');
    graphTab.classList.add('locked');
  }
  
  // Update feature buttons in features tab
  const featureButtons = document.querySelectorAll('.feature-btn');
  featureButtons.forEach(btn => {
    const feature = btn.closest('.premium-feature');
    const featureName = feature.querySelector('h3').textContent;
    
    if (
      (featureName.includes('Knowledge Graph') || featureName.includes('Cross-Video') || featureName.includes('Real-Time')) && currentPlanLevel >= 2
    ) {
      btn.classList.remove('locked');
      btn.textContent = 'Activate';
      feature.classList.remove('locked');
    } else if (currentPlanLevel < 2) {
      btn.classList.add('locked');
      feature.classList.add('locked');
    }
  });
}

async function checkVideoStatus(videoId) {
  try {
    // Check if video was already processed by calling the backend
    const response = await fetch(`${API_BASE_URL}/videos`);
    if (response.ok) {
      const data = await response.json();
      const processedVideo = data.processed_videos.find(v => v.video_id === videoId);
      
      if (processedVideo) {
        isProcessed = true;
        updateStatus('Video processed - Chat ready!', 'processing');
        enableChat();
        showSmartSuggestions();
      } else {
        // Auto-process the video
        await processCurrentVideo();
      }
    } else {
      // If backend is not available, try to process
      await processCurrentVideo();
    }
  } catch (error) {
    console.error('Error checking video status:', error);
    updateStatus('Backend not available. Please start the server.', 'error');
  }
}

async function processCurrentVideo() {
  if (!currentVideoId || isProcessing) return;
  
  isProcessing = true;
  updateStatus('Processing video...', 'processing');
  showLoading(true);
  
  try {
    const response = await fetch(`${API_BASE_URL}/process_video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        video_id: currentVideoId,
        user_id: currentUserId
      }),
    });
    
    if (response.ok) {
      const data = await response.json();
      isProcessed = true;
      
      // Add system message
      addMessage('system', data.message);
      
      updateStatus('Video processed - Chat enabled!', 'processing');
      enableChat();
      showSmartSuggestions();
      
      // Update subscription info if provided
      if (data.subscription_info) {
        updateSubscriptionUI(data.subscription_info, {});
      }
      
      // Store processing status
      await chrome.storage.local.set({ 
        [`processed_${currentVideoId}`]: true,
        [`messages_${currentVideoId}`]: await getStoredMessages(currentVideoId)
      });
      
    } else {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process video');
    }
  } catch (error) {
    console.error('Error processing video:', error);
    addMessage('error', `Error: ${error.message}`);
    updateStatus('Processing failed. Check if backend is running.', 'error');
  } finally {
    isProcessing = false;
    showLoading(false);
  }
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message || !isProcessed || !currentVideoId) return;
  
  // Check if user has chat access
  if (userSubscription.plan === 'free') {
    const chatUsage = userSubscription.features_used_today?.basic_chat || 0;
    if (chatUsage >= 50) {
      showUpgradePrompt('Daily chat limit reached! Upgrade for unlimited chats.');
      return;
    }
  }
  
  // Add user message
  addMessage('user', message);
  messageInput.value = '';
  
  // Show loading
  showLoading(true);
  disableInput();
  
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        video_id: currentVideoId,
        query: message,
        user_id: currentUserId
      }),
    });
    
    if (response.ok) {
      const data = await response.json();
      let responseText = data.response || 'No response received';
      
      // Format with markdown
      const formattedResponse = marked.parse(responseText);
      addMessage('assistant', formattedResponse);
      
      // Handle premium features in response
      if (data.premium_features) {
        handlePremiumFeaturesResponse(data.premium_features);
      }
      
      // Show upgrade prompt if provided
      if (data.upgrade_prompt) {
        showUpgradePrompt(data.upgrade_prompt.message);
      }
      
      // Update subscription info
      if (data.subscription_info) {
        await loadUserSubscription(); // Refresh subscription data
      }
      
    } else {
      const errorData = await response.json();
      
      // Handle subscription errors
      if (response.status === 403 && errorData.detail?.error === 'Premium feature access required') {
        showUpgradePrompt(errorData.detail.upgrade_message);
        return;
      } else if (response.status === 403 && errorData.detail?.error === 'Daily chat limit exceeded') {
        showUpgradePrompt(errorData.detail.upgrade_message);
        return;
      }
      
      throw new Error(errorData.detail || 'Failed to get response');
    }
  } catch (error) {
    console.error('Error sending message:', error);
    addMessage('error', `Error: ${error.message}`);
  } finally {
    showLoading(false);
    enableInput();
    
    // Store updated messages
    if (currentVideoId) {
      await chrome.storage.local.set({
        [`messages_${currentVideoId}`]: await getStoredMessages(currentVideoId)
      });
    }
  }
}

function handlePremiumFeaturesResponse(premiumFeatures) {
  // Handle brain mode active indicator
  if (premiumFeatures.brain_mode_active) {
    addMessage('system', '🧠 Brain Mode: Personalized response generated');
  }
  
  // Handle follow-up suggestions
  if (premiumFeatures.follow_up_suggestions) {
    showFollowUpSuggestions(premiumFeatures.follow_up_suggestions);
  }
  
  // Handle timeline response
  if (premiumFeatures.timeline_response) {
    addMessage('system', '⏱️ Timeline-aware response generated');
  }
  
  // Handle visual analysis
  if (premiumFeatures.visual_response) {
    addMessage('system', '👁️ Visual content analysis completed');
  }
}

function showFollowUpSuggestions(suggestions) {
  const suggestionsContainer = document.getElementById('suggestions');
  if (suggestions && suggestions.length > 0) {
    suggestionsContainer.innerHTML = '';
    suggestions.slice(0, 3).forEach(suggestion => {
      const item = document.createElement('div');
      item.className = 'suggestion-item';
      item.textContent = `💡 ${suggestion}`;
      item.addEventListener('click', () => {
        messageInput.value = suggestion;
        sendMessage();
      });
      suggestionsContainer.appendChild(item);
    });
    suggestionsContainer.classList.add('show');
  }
}

function showUpgradePrompt(message) {
  const upgradeMessage = document.createElement('div');
  upgradeMessage.className = 'message upgrade-prompt';
  upgradeMessage.innerHTML = message;
  upgradeMessage.addEventListener('click', () => {
    showUpgradeModal();
  });
  
  messagesContainer.appendChild(upgradeMessage);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showUpgradeModal() {
  upgradeModal.classList.add('show');
}

function hideUpgradeModal() {
  upgradeModal.classList.remove('show');
}

async function startTrialSubscription() {
  try {
    const response = await fetch(`${API_BASE_URL}/subscription/trial`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: currentUserId,
        plan: selectedPlan,
        trial_days: 7
      }),
    });
    
    if (response.ok) {
      const data = await response.json();
      addMessage('system', `🎉 ${data.message}! All premium features are now available.`);
      
      // Refresh subscription data
      await loadUserSubscription();
      
      hideUpgradeModal();
      
      // Show success message
      updateStatus('Trial started successfully!', 'processing');
      
    } else {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start trial');
    }
  } catch (error) {
    console.error('Error starting trial:', error);
    addMessage('error', `Trial error: ${error.message}`);
  }
}

// Premium feature functions
async function toggleBrainMode() {
  if (userSubscription.plan === 'free') {
    showUpgradePrompt('🧠 Upgrade to Pro for Brain Mode! Get personalized responses that learn from your interactions.');
    return;
  }
  
  brainModeBtn.classList.toggle('active');
  const isActive = brainModeBtn.classList.contains('active');
  
  if (isActive) {
    addMessage('system', '🧠 Brain Mode activated - Personalized responses enabled');
  } else {
    addMessage('system', '🧠 Brain Mode deactivated');
  }
}

async function toggleTimelineMode() {
  if (userSubscription.plan === 'free') {
    showUpgradePrompt('⏱️ Upgrade to Pro for Timeline-Aware conversations! Ask about specific video moments.');
    return;
  }
  
  timelineBtn.classList.toggle('active');
  const isActive = timelineBtn.classList.contains('active');
  
  if (isActive) {
    addMessage('system', '⏱️ Timeline Mode activated - Ask about specific timestamps');
    messageInput.placeholder = 'Ask about specific times (e.g., "What happens at 2:30?")';
  } else {
    addMessage('system', '⏱️ Timeline Mode deactivated');
    messageInput.placeholder = 'Ask about the video...';
  }
}

// Tab system
function setupTabSystem() {
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Check if tab is locked
      if (tab.classList.contains('locked')) {
        const tabName = tab.dataset.tab;
        let requiredPlan = 'Pro';
        
        if (tabName === 'features' || tabName === 'graph') {
          requiredPlan = 'Power+';
        }
        
        showUpgradePrompt(`🔒 Upgrade to ${requiredPlan} to unlock this feature!`);
        return;
      }
      
      const targetTab = tab.dataset.tab;
      
      // Update active tab
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      
      // Update active content
      tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${targetTab}-tab`) {
          content.classList.add('active');
        }
      });
    });
  });
}

function setupUpgradeModal() {
  // Plan selection
  const planOptions = document.querySelectorAll('.plan-option');
  planOptions.forEach(option => {
    option.addEventListener('click', () => {
      planOptions.forEach(p => p.classList.remove('selected'));
      option.classList.add('selected');
      selectedPlan = option.dataset.plan;
    });
  });
  
  // Modal buttons
  closeModal.addEventListener('click', hideUpgradeModal);
  startTrial.addEventListener('click', startTrialSubscription);
  
  // Close modal on outside click
  upgradeModal.addEventListener('click', (e) => {
    if (e.target === upgradeModal) {
      hideUpgradeModal();
    }
  });
}

function setupEventListeners() {
  // Send message on button click
  sendButton.addEventListener('click', sendMessage);
  
  // Send message on Enter key (but allow Shift+Enter for new line)
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  // Auto-resize textarea
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 80) + 'px';
  });
  
  // Premium feature buttons
  brainModeBtn.addEventListener('click', toggleBrainMode);
  timelineBtn.addEventListener('click', toggleTimelineMode);
  upgradeBtn.addEventListener('click', showUpgradeModal);
  
  // Smart suggestions
  document.getElementById('suggestions').addEventListener('click', (e) => {
    if (e.target.classList.contains('suggestion-item')) {
      const suggestion = e.target.textContent.replace('💡 ', '');
      messageInput.value = suggestion;
      sendMessage();
    }
  });
}

// Utility functions
function addMessage(type, content) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  
  if (type === 'assistant') {
    messageDiv.innerHTML = content;
  } else {
    messageDiv.textContent = content;
  }
  
  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateStatus(message, type = 'info') {
  statusBar.textContent = message;
  statusBar.className = `status-bar ${type}`;
}

function showLoading(show) {
  if (show) {
    loading.classList.add('show');
  } else {
    loading.classList.remove('show');
  }
}

function enableChat() {
  messageInput.disabled = false;
  sendButton.disabled = false;
  messageInput.placeholder = 'Ask about the video...';
}

function disableInput() {
  messageInput.disabled = true;
  sendButton.disabled = true;
}

function enableInput() {
  messageInput.disabled = false;
  sendButton.disabled = false;
}

function showSmartSuggestions() {
  const suggestionsContainer = document.getElementById('suggestions');
  suggestionsContainer.classList.add('show');
}

async function loadStoredMessages(videoId) {
  try {
    const result = await chrome.storage.local.get([`messages_${videoId}`]);
    const storedMessages = result[`messages_${videoId}`];
    
    if (storedMessages && storedMessages.length > 0) {
      storedMessages.forEach(msg => {
        addMessage(msg.type, msg.content);
      });
    }
  } catch (error) {
    console.error('Error loading stored messages:', error);
  }
}

async function getStoredMessages(videoId) {
  const messages = [];
  const messageElements = messagesContainer.querySelectorAll('.message:not(.welcome)');
  
  messageElements.forEach(element => {
    const type = Array.from(element.classList).find(cls => 
      ['user', 'assistant', 'system', 'error'].includes(cls)
    );
    
    if (type) {
      messages.push({
        type: type,
        content: element.innerHTML || element.textContent
      });
    }
  });
  
  return messages;
}

