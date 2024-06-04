Chatbot Knowledge Extraction and Interaction
Overview

This chatbot is designed to extract information from URLs provided by users, save the extracted information to a .txt file, and index the file for efficient chat interactions. By continuously updating the knowledge base with each new URL, the chatbot ensures persistent knowledge retention, overcoming the limitations of context windows seen in traditional models like ChatGPT. This means you won't need to repeatedly paste links or copy old chat history.
Features

    URL Information Extraction: Send a URL to the chatbot, and it will extract the relevant information.
    Persistent Knowledge Base: Extracted information is saved and indexed, ensuring it is retained for future interactions.
    Efficient Interaction: Indexed knowledge allows for fast and accurate responses during chat interactions.

Getting Started

To use this repo, follow these steps:

    Clone the Repository:

    git clone https://github.com/maerlic/AdvancedRAGChatbotPlusWebScraperJavaScript.git

Navigate to the Project Folder:

    cd AdvancedRAGChatbotPlusWebScraperJavaScript

Install Dependencies:

    npm install

Create an .env File:
Create a file named .env in the project root and add the following keys:

.env

    ANTHROPIC_API_KEY=your_anthropic_api_key
    OPENAI_API_KEY=your_openai_api_key

Replace your_anthropic_api_key and your_openai_api_key with your actual API keys.

Hugging Face Account:
Ensure you have a Hugging Face account to use the embedding model WhereIsAI/UAE-Large-V1.

Running the Chatbot

    Start the Server:
    Open a terminal and run:

    node server.mjs

Launch the Chatbot UI:
    Open another terminal and run:

    npm start

Notes

    The code is still under development.
    The indexing process currently takes longer than desired, and improvements are ongoing.

Contributing

Feel free to submit issues or pull requests. Your contributions are welcome!
