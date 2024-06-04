import { AutoTokenizer, AutoModel } from '@xenova/transformers';
import * as math from 'mathjs';
import { BaseEmbedding } from 'llamaindex';

class LocalEmbedding extends BaseEmbedding {
  constructor(modelName = 'WhereIsAI/UAE-Large-V1', maxLength = 512) {
    super();
    this.tokenizer = null;
    this.model = null;
    this.modelName = modelName;
    this.maxLength = maxLength;
  }

  async init() {
    try {
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelName);
      this.model = await AutoModel.from_pretrained(this.modelName);
    } catch (error) {
      console.error('Error initializing model or tokenizer:', error);
      throw error;
    }
  }

  async getTextEmbeddingBatch(texts) {
    const embeddings = [];
    for (const text of texts) {
      try {
        // Ensure the text is a string
        const inputText = typeof text === 'string' ? text : JSON.stringify(text);

        // Pad or truncate the input text to the specified maximum length
        const inputs = await this.tokenizer(inputText, {
          return_tensors: true,
          truncation: true,
          padding: 'max_length',
          max_length: this.maxLength,
        });

        console.log("Model: ", this.model);
        const outputs = await this.model(inputs);
        console.log("Model outputs: ", outputs);
        const embedding = outputs.last_hidden_state.data;
        embeddings.push(embedding);
      } catch (error) {
        console.error(`Error processing text: "${text}"`, error);
        throw error;
      }
    }
    return embeddings;
  }

  async getAggEmbeddingFromQueries(queries) {
    try {
      const embeddings = await this.getTextEmbeddingBatch(queries);
      const aggEmbedding = math.mean(embeddings, 0);
      return aggEmbedding;
    } catch (error) {
      console.error('Error aggregating embeddings:', error);
      throw error;
    }
  }

  async transform(texts) {
    return this.getTextEmbeddingBatch(texts);
  }

  async getQueryEmbedding(query) {
    const [embedding] = await this.getTextEmbeddingBatch([query]);
    return embedding;
  }
}

export default LocalEmbedding;