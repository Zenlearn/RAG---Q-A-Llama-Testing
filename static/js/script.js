// scripts.js
document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent form submission
        sendMessage();
    }
});

// Dynamic placeholder phrases
const placeholderPhrases = [
    "Ask me anything...",
    "Need help?",
    "What can I do for you?",
    "How can I assist you today?",
    "Type your question here...",
];

// Function to cycle through placeholder text
function changePlaceholder() {
    const inputField = document.getElementById('user-input');
    let index = 0;
    setInterval(() => {
        index = (index + 1) % placeholderPhrases.length;
        inputField.setAttribute('placeholder', placeholderPhrases[index]);
    }, 5000); // Change every 5 seconds
}

// Auto-expand the input field when typing long messages
function autoExpandInputField() {
    const inputField = document.getElementById('user-input');
    inputField.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// Toggle between dark and light theme and change the theme icon
document.getElementById('theme-toggle-button').addEventListener('click', function() {
    document.body.classList.toggle('dark-theme');
    const currentTheme = document.body.classList.contains('dark-theme') ? 'dark' : 'light';
    localStorage.setItem('theme', currentTheme);
    // Change the theme icon (moon for dark mode, sun for light mode)
    const themeIcon = document.getElementById('theme-icon');
    if (currentTheme === 'dark') {
        themeIcon.src = 'sun-icon.png';  // Ensure this image exists in /static/images/
    } else {
        themeIcon.src = 'moon-icon.png';
    }
});

// On page load, apply saved theme and load conversation history
window.addEventListener('load', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme && savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        document.getElementById('theme-icon').src = 'sun-icon.png';  // Switch to sun icon for dark theme
    }
    // Load saved conversation history
    const savedHistory = JSON.parse(localStorage.getItem('conversationHistory'));
    if (savedHistory) {
        conversationHistory = savedHistory;
        conversationHistory.forEach(msg => {
            addMessage(msg.content, msg.role === 'user' ? 'user-message' : 'bot-message');
        });
    }

    changePlaceholder(); // Start cycling placeholder text
    autoExpandInputField(); // Enable auto-expansion for input field
});

// Array to track the conversation history
let conversationHistory = [];

// Sample quick replies
const quickReplies = [
    "Can you tell me more?",
    "What are the next steps?",
    "I'm interested!",
    "Thank you!",
    "I need more details."
];

// Generate quick reply buttons
function generateQuickReplies() {
    const quickRepliesContainer = document.getElementById('quick-replies-container');
    quickRepliesContainer.innerHTML = ''; // Clear previous replies
    quickReplies.forEach(reply => {
        const button = document.createElement('button');
        button.classList.add('quick-reply-button');
        button.innerText = reply;
        button.addEventListener('click', () => {
            document.getElementById('user-input').value = reply;
            sendMessage();
        });
        quickRepliesContainer.appendChild(button);
    });
}

// Send message function
async function sendMessage() {
    const userInput = document.getElementById('user-input').value.trim();
    if (!userInput) {
        alert("Please enter a message");
        return;
    }

    addMessage(userInput, 'user-message');
    const quickRepliesContainer = document.getElementById('quick-replies-container');
    quickRepliesContainer.innerHTML = '';
    conversationHistory.push({ role: "user", content: userInput });
    localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));

    const chatWindow = document.getElementById('chat-window');
    chatWindow.scrollTop = chatWindow.scrollHeight;
    showTypingIndicator();

    try {
        const backendUrl = 'http://13.200.228.207:8080/ask_pdf';

        const response = await fetch(backendUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Origin': window.location.origin
            },
            body: JSON.stringify({ query: userInput }),
            credentials: 'include'
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        const botMessage = result.answer;

        hideTypingIndicator();
        addMessage(botMessage, 'bot-message');
        conversationHistory.push({ role: "assistant", content: botMessage });
        localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));

        document.getElementById('user-input').value = '';
        chatWindow.scrollTop = chatWindow.scrollHeight;
        generateQuickReplies();
    } catch (error) {
        console.error("Error:", error);
        hideTypingIndicator();
        addMessage("Sorry, there was an error processing your request. Please try again later.", 'bot-message');
    }
}

// Add messages to the chat window
function addMessage(message, className) {
    const chatWindow = document.getElementById('chat-window');
    const wrapper = document.createElement('div');
    wrapper.classList.add('message-wrapper');
    const isUserMessage = className === 'user-message';
    if (isUserMessage) {
        wrapper.classList.add('user-message-wrapper');
    } else {
        wrapper.classList.add('bot-message-wrapper');
    }
    const avatar = document.createElement('img');
    avatar.src = isUserMessage ? '/static/images/corporate-user-icon.png' : '/static/images/Acharya.ai-Finance for growth.png';
    avatar.classList.add('avatar');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', className, 'fade-in');
    messageDiv.innerText = message;
    wrapper.appendChild(avatar);
    wrapper.appendChild(messageDiv);
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showTypingIndicator() {
    const chatWindow = document.getElementById('chat-window');
    const typingIndicatorWrapper = document.createElement('div');
    typingIndicatorWrapper.classList.add('message-wrapper', 'bot-message-wrapper');
    typingIndicatorWrapper.id = 'typing-indicator-wrapper';
    const avatar = document.createElement('img');
    avatar.src = '/static/images/Acharya.ai-Finance for growth.png';
    avatar.classList.add('avatar');
    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('chat-message', 'bot-message', 'typing-indicator');
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';  // Simple loading animation
    typingIndicatorWrapper.appendChild(avatar);
    typingIndicatorWrapper.appendChild(typingIndicator);
    chatWindow.appendChild(typingIndicatorWrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicatorWrapper = document.getElementById('typing-indicator-wrapper');
    if (typingIndicatorWrapper) {
        typingIndicatorWrapper.remove();
    }
}

// Save conversation history before the window is closed or refreshed
window.addEventListener('beforeunload', () => {
    localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
});