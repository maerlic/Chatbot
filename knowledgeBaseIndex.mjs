import fs from 'fs';
import path from 'path';
import {
    VectorStoreIndex,
    TextNode,
    SimpleVectorStore,
    storageContextFromDefaults,
    serviceContextFromDefaults,
    jsonToNode,
    ObjectType,
    PromptHelper
} from 'llamaindex';

class KnowledgeBaseIndexer {
    constructor(filePath, embeddingModel, deleteExistingIndex = false) {
        this.filePath = filePath;
        this.embeddingModel = embeddingModel;
        this.deleteExistingIndex = deleteExistingIndex;
        this.index = null;
        this.retriever = null; // Add retriever property
    }

    async init() {
        try {
            await this.embeddingModel.init();
            await this.loadOrCreateIndex();
        } catch (error) {
            console.error('Error initializing KnowledgeBaseIndexer:', error);
            throw error;
        }
    }

    async loadOrCreateIndex() {
        const persistDir = 'index_dir';
        if (this.deleteExistingIndex && fs.existsSync(persistDir)) {
            fs.rmdirSync(persistDir, { recursive: true });
        }
        fs.mkdirSync(persistDir, { recursive: true });

        const indexPath = path.join(persistDir, 'index.json');
        if (fs.existsSync(indexPath)) {
            const indexData = await fs.promises.readFile(indexPath, 'utf-8');
            const { nodes, vectorStoreData } = JSON.parse(indexData);
            const vectorStore = SimpleVectorStore.fromDict(vectorStoreData, this.embeddingModel);

            console.log('Initializing VectorStoreIndex with nodes and vector store from file...');
            this.index = await VectorStoreIndex.init({
                nodes: nodes.map(node => jsonToNode(node, ObjectType.TEXT)),
                vectorStores: { TEXT: vectorStore },
                serviceContext: this.createServiceContext(),
                storageContext: await storageContextFromDefaults({ persistDir }),
            });
            console.log('VectorStoreIndex initialized from file:', this.index);

            // Initialize retriever
            this.retriever = this.index.asRetriever({ similarityTopK: 3 });
        } else {
            await this.indexDocuments();
        }
    }

    async loadDocuments() {
        const content = await fs.promises.readFile(this.filePath, 'utf-8');
        const chunkSize = 75; // Define chunk size (number of words per chunk)
        const chunks = this.chunkText(content, chunkSize);
        console.log("Chunks: ", chunks)
        return chunks.map((chunk, index) => ({
            text: chunk,
            title: `${path.basename(this.filePath, path.extname(this.filePath))}_part${index + 1}`,
            url: `${this.filePath}#part${index + 1}`
        }));
    }

    chunkText(text, chunkSize) {
        const words = text.split(/\s+/);
        let chunks = [];
        
        for (let i = 0; i < words.length; i += chunkSize) {
            chunks.push(words.slice(i, i + chunkSize).join(' '));
        }
        
        return chunks;
    }

    createNodesFromDocuments(documents, embeddings) {
        return documents.map((doc, i) => new TextNode({
            id_: `doc_${i}`,
            text: doc.text,
            metadata: { title: doc.title, url: doc.url },
            embedding: embeddings[i]
        }));
    }

    async indexDocuments() {
        try {
            const documents = await this.loadDocuments();
            const texts = documents.map(doc => doc.text);
            const embeddings = await this.embeddingModel.getTextEmbeddingBatch(texts);
            console.log("Embeddings: ", embeddings)
            const nodes = this.createNodesFromDocuments(documents, embeddings);

            const vectorStore = new SimpleVectorStore({
                data: { embeddingDict: {}, textIdToRefDocId: {} },
                embedModel: this.embeddingModel,
                storesText: true,
            });

            const storageContext = await storageContextFromDefaults({
                persistDir: 'index_dir',
                vectorStoreClass: SimpleVectorStore,
                embedModel: this.embeddingModel,
            });

            const serviceContext = this.createServiceContext();

            console.log('Storage Context:', storageContext);
            console.log('Vector Stores:', storageContext.vectorStores);

            if (!storageContext.vectorStores.TEXT) {
                throw new Error('TEXT vector store is not initialized properly.');
            }

            console.log('Initializing VectorStoreIndex with nodes and vector stores...');
            this.index = await VectorStoreIndex.init({
                nodes: nodes,
                vectorStores: { TEXT: vectorStore },
                serviceContext: serviceContext,
                storageContext: storageContext,
            });
            console.log('Index initialized successfully:', this.index);

            // Initialize retriever
            this.retriever = this.index.asRetriever({ similarityTopK: 3 });

            const serializedIndex = JSON.stringify({
                nodes: nodes.map(node => node.toJSON()),
                vectorStoreData: vectorStore.toDict()
            });

            await fs.promises.writeFile('index_dir/index.json', serializedIndex);
            console.log('Documents indexed successfully');
        } catch (error) {
            console.error('Error indexing documents:', error);
            throw error;
        }
    }

    createServiceContext() {
        return serviceContextFromDefaults({
            embedModel: this.embeddingModel
        });
    }

    async query(userPrompt) {
        try {
            if (!this.retriever) {
                throw new Error('Retriever is not initialized');
            }
            
            // Get the maximum token length from the indexed documents
            const maxTokenLength = this.index.indexStruct.nodesDict.doc_0.getContent().split(' ').length;
            
            // Truncate or pad the user prompt to match the maximum token length
            const paddedUserPrompt = userPrompt.split(' ').slice(0, maxTokenLength).join(' ');
            
            const results = await this.retriever.retrieve({ query: paddedUserPrompt });
            const topResults = results.slice(0, 1);
    
            return topResults.map(result => result.node.getContent()).join('\n\n');
        } catch (error) {
            console.error('Error querying the index:', error);
            throw error;
        }
    }
}

export default KnowledgeBaseIndexer;
