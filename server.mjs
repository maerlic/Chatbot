import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import { Anthropic } from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import { promises as fs } from 'fs';
import LocalEmbedding from './LocalEmbedding.mjs';
import KnowledgeBaseIndexer from './knowledgeBaseIndex.mjs';

dotenv.config();

puppeteer.use(StealthPlugin());

const app = express();
const port1 = process.env.PORT || 5000;
const port2 = 5001;
const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
const anthropic = new Anthropic(anthropicApiKey);

app.use(bodyParser.json());
app.use(cors());

const timeout = 8000;

// Ensure the ANTHROPIC_API_KEY is set
if (!process.env.ANTHROPIC_API_KEY) {
  throw new Error("Anthropic API key not set in environment variables. Set the ANTHROPIC_API_KEY environment variable.");
}

const filePath = 'knowledge_base.txt'; // Change this to your actual file path
const embeddingModel = new LocalEmbedding();

const knowledgeBaseIndexer = new KnowledgeBaseIndexer(filePath, embeddingModel);

async function image_to_base64(imageFile) {
  try {
    const data = await fs.readFile(imageFile);
    const base64Data = data.toString('base64').replace(/\r?\n|\r/g, '');
    return base64Data;
  } catch (err) {
    console.error('Error reading the file:', err);
    throw err;
  }
}

(async () => {
  try {
      await knowledgeBaseIndexer.init();
      //await knowledgeBaseIndexer.indexDocuments(); // Ensure the knowledge base is indexed upon server initialization
  } catch (error) {
      console.error('Failed to initialize the knowledge base indexer:', error);
      process.exit(1); // Exit the application if initialization fails
  }

  app.post('/chat', async (req, res) => {
      const userPrompt = req.body.message;
      const url = extractUrl(userPrompt);
      let message_text = "No response";

      if (url) {
          const browser = await puppeteer.launch({ headless: "new" });
          const page = await browser.newPage();
          await page.setViewport({ width: 1200, height: 1600, deviceScaleFactor: 1.75 });

          const systemMessage = `You are a website crawler. You will be given instructions on what to do by browsing. You are connected to a web browser and you will be given the screenshot of the website you are on. The links on the website will be highlighted in red in the screenshot. Always read what is in the screenshot. Don't guess link names.
          You can go to a specific URL by answering with the following JSON format:
          {"url": "url goes here"}
          You can click links on the website by referencing the text inside of the link/button, by answering in the following JSON format:
          {"click": "Text in link"}
          Once you are on a URL and you have found the answer to the user's question, you can answer with a regular message.
          In the beginning, go to a direct URL that you think might contain the answer to the user's question. Prefer to go directly to sub-urls like 'https://google.com/search?q=search' if applicable. Prefer to use Google for simple queries. If the user provides a direct URL, go to that one.`;

          let screenshot_taken = false;
          let base64_image;

          if (url) {
              await page.goto(url, { waitUntil: "domcontentloaded" });
              await highlight_links(page);
              await Promise.race([waitForEvent(page, 'load'), sleep(10000)]); // Assuming 10 seconds as the timeout
              await highlight_links(page);
              await page.screenshot({ path: "screenshot.jpg", quality: 100 });
              screenshot_taken = true;
          }

          if (screenshot_taken) {
              base64_image = await image_to_base64("screenshot.jpg");
          }

          if (base64_image) {
              const response = await anthropic.messages.create({
                  model: "claude-3-opus-20240229",
                  max_tokens: 1024,
                  system: systemMessage,
                  messages: [
                      {
                          role: 'user',
                          content: [
                              {
                                  type: 'text',
                                  text: userPrompt
                              },
                              {
                                  type: 'image',
                                  source: {
                                      type: 'base64',
                                      media_type: 'image/jpeg',
                                      data: base64_image
                                  }
                              }
                          ]
                      }
                  ]
              });

              message_text = response.content[0].text;
              message_text = message_text.replace(/\\n/g, '\n');

              try {
                  await fs.appendFile('knowledge_base.txt', message_text + '\n');
                  console.log('Response appended to knowledge_base.txt');
                  await knowledgeBaseIndexer.indexDocuments(); // Update the index with the new data
              } catch (err) {
                  console.error('Error writing to file:', err);
              }
          }

          await browser.close();
        } else {
          console.log("Entering else block");
          try {
              // Use the indexed documents to answer the question
              console.log("Querying the knowledge base with user prompt:", userPrompt);
              const kbResponse = await knowledgeBaseIndexer.query(userPrompt);
              console.log("Knowledge base response received:", kbResponse);
              
              const combinedPrompt = `The following is information from the knowledge base that might help answer the question: "${kbResponse}". Now, answer the user's question based on this information: "${userPrompt}"`;
              console.log("Combined prompt created:", combinedPrompt);
      
              console.log("Sending request to Anthropics API");
              const response = await anthropic.messages.create({
                  model: "claude-3-haiku-20240307",
                  max_tokens: 1024,
                  system: "You are an intelligent assistant.",
                  messages: [
                      {
                          role: 'user',
                          content: combinedPrompt
                      }
                  ]
              });
              console.log("Response received from Anthropics API:", response);
      
              message_text = response.content[0].text;
              console.log("Message text extracted from response:", message_text);
          } catch (error) {
              console.error("Error in else block:", error);
              res.status(500).send({ error: 'An error occurred while processing your request.' });
              return;
          }
      }      
    
    res.send({ role: 'assistant', content: message_text });
  });
    
  app.listen(port1, () => {console.log(`Server for Claude 3 model is running on http://localhost:${port1}`);});
    
})();
    

// Helper functions for extracting URL, sleeping, highlighting links, and waiting for events
function extractUrl(text) {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const matches = text.match(urlRegex);
  return matches ? matches[0] : null;
}

async function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function highlight_links(page) {
  await page.evaluate(() => {
    document.querySelectorAll('[claude-link-text]').forEach(e => {
      e.removeAttribute("claude-link-text");
    });
  });

  const elements = await page.$$(
    "a, button, input, textarea, [role=button], [role=treeitem]"
  );

  elements.forEach(async e => {
    await page.evaluate(e => {
      function isStyleVisible(el) {
        const style = window.getComputedStyle(el);
        return style.width !== '0' &&
          style.height !== '0' &&
          style.opacity !== '0' &&
          style.display !== 'none' &&
          style.visibility !== 'hidden';
      }

      function isElementInViewport(el) {
        const rect = el.getBoundingClientRect();
        return (
          rect.top >= 0 &&
          rect.left >= 0 &&
          rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
          rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
      }

      function isElementVisible(el) {
        if (!el) return false;

        if (!isStyleVisible(el)) {
          return false;
        }

        let parent = el;
        while (parent) {
          if (!isStyleVisible(parent)) {
            return false;
          }
          parent = parent.parentElement;
        }

        return isElementInViewport(el);
      }

      e.style.border = "1px solid red";

      const position = e.getBoundingClientRect();

      if (position.width > 5 && position.height > 5 && isElementVisible(e)) {
        const link_text = e.textContent.replace(/[^a-zA-Z0-9 ]/g, '');
        e.setAttribute("claude-link-text", link_text);
      }
    }, e);
  });
}

async function waitForEvent(page, event) {
  return page.evaluate(event => {
    return new Promise((resolve) => {
      document.addEventListener(event, function () {
        resolve();
      });
    });
  }, event);
}
