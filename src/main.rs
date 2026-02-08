use rig::client::{CompletionClient, Nothing};
use rig::providers::ollama::{self, EmbeddingModel};
use rig::embeddings::EmbeddingsBuilder;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::completion::Prompt;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 1. Initialize Ollama Client
    // In 0.3.0, if you use the default local address, you can just use default()
    // let client = ollama::Client::from("http://localhost:11434");
    let client: ollama::Client = ollama::Client::builder()
        .api_key(Nothing)
        .build()
        .unwrap();

    // 2. Load PDF and Extract Text
    println!("📖 Reading knowledge.pdf...");
    let pdf_path: &Path = Path::new("knowledge.pdf");
    
    // Ensure the file exists before trying to read it
    if !pdf_path.exists() {
        anyhow::bail!("File 'knowledge.pdf' not found! Please place it in the project folder.");
    }
    
    let content: String = pdf_extract::extract_text(&pdf_path)?;

    // 3. Create Embeddings Model
    // let embedding_model = client.embedding_model(ollama::NOMIC_EMBED_TEXT);
    println!("📖 create embedding model");
    // let embedding_model: ollama::EmbeddingModel = client.embedding_model(ollama::NOMIC_EMBED_TEXT, 768);
    let embedding_model: EmbeddingModel = ollama::EmbeddingModel::new(
        client.clone(),
    ollama::NOMIC_EMBED_TEXT,
        768  // dimensions for nomic-embed-text
    );
    
    println!("📖 created embedding model");

    // 4. Build the Vector Store
    println!("🔨 Indexing document (M2 Pro Power)...");
    let embeddings: Vec<(String, rig::OneOrMany<rig::embeddings::Embedding>)> = EmbeddingsBuilder::new(embedding_model.clone())
        .document(content.clone())? // Note the '?' for error handling in 0.3.0
        .build()
        .await?;

    let vector_store: InMemoryVectorStore<String>    = InMemoryVectorStore::from_documents(embeddings);

    // 5. Create the RAG Agent
    // We use the vector store as a 'dynamic context'
    let perso_agent = client.agent("llama3:latest")
        .preamble("You are 'Perso', a personal assistant. Use the provided context to answer questions accurately.")
        .dynamic_context(2, vector_store.index(embedding_model)) 
        .build();

    // 6. Interactive Chat Loop
    println!("✨ Perso is ready! (Type 'exit' to quit)");
    
    loop {
        let mut input: String = String::new();
        print!("\n👤 You: ");
        use std::io::Write; // Explicitly bring trait into scope
        std::io::stdout().flush()?;
        std::io::stdin().read_line(&mut input)?;

        let trim_input = input.trim();
        if trim_input == "exit" || trim_input == "quit" { break; }
        if trim_input.is_empty() { continue; }

        println!("🤖 Perso thinking...");
        let response = perso_agent.prompt(trim_input).await?;
        println!("🤖 Perso: {}", response);
    }

    Ok(())
}