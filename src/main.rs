use anyhow::{Context, Result};
use rig::client::{CompletionClient, Nothing};
use rig::completion::Prompt;
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::ollama::{self, EmbeddingModel};
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use std::io::{self, Write};
use std::path::Path;

const PDF_PATH: &str = "knowledge.pdf";
const EMBEDDING_MODEL: &str = ollama::NOMIC_EMBED_TEXT;
const EMBEDDING_DIMS: usize = 768;
const LLM_MODEL: &str = "llama3:latest";
const TOP_K_RESULTS: usize = 3; // Increased from 2 for better context

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize Ollama Client
    let client: rig::client::Client<ollama::OllamaExt> = create_ollama_client()?;

    // 2. Load and process PDF
    println!("📖 Reading {}...", PDF_PATH);
    let content: String = load_pdf_content(PDF_PATH)?;

    // 3. Create embeddings and vector store
    println!("🔨 Creating embeddings...");
    let embedding_model: EmbeddingModel = create_embedding_model(&client);
    let vector_store: InMemoryVectorStore<String> =
        build_vector_store(content, &embedding_model).await?;

    // 4. Create RAG agent
    let agent: rig::agent::Agent<ollama::CompletionModel> = client
        .agent(LLM_MODEL)
        .preamble(
            "You are 'Perso', a knowledgeable personal assistant. \
             Answer questions accurately based on the provided context. \
             If the context doesn't contain relevant information, say so honestly.",
        )
        .dynamic_context(TOP_K_RESULTS, vector_store.index(embedding_model))
        .build();

    // 5. Interactive chat loop
    run_chat_loop(agent).await?;

    Ok(())
}

fn create_ollama_client() -> Result<ollama::Client> {
    ollama::Client::builder()
        .api_key(Nothing)
        .build()
        .context("Failed to create Ollama client")
}

fn load_pdf_content(path: &str) -> Result<String> {
    let pdf_path: &Path = Path::new(path);

    if !pdf_path.exists() {
        anyhow::bail!(
            "File '{}' not found! Please place it in the project folder.",
            path
        );
    }

    pdf_extract::extract_text(pdf_path).context("Failed to extract text from PDF")
}

fn create_embedding_model(client: &ollama::Client) -> EmbeddingModel {
    EmbeddingModel::new(client.clone(), EMBEDDING_MODEL, EMBEDDING_DIMS)
}

async fn build_vector_store(
    content: String,
    embedding_model: &EmbeddingModel,
) -> Result<InMemoryVectorStore<String>> {
    let embeddings: Vec<(String, rig::OneOrMany<rig::embeddings::Embedding>)> =
        EmbeddingsBuilder::new(embedding_model.clone())
            .document(content)
            .context("Failed to create document embedding")?
            .build()
            .await
            .context("Failed to build embeddings")?;

    Ok(InMemoryVectorStore::from_documents(embeddings))
}

async fn run_chat_loop(agent: impl Prompt) -> Result<()> {
    println!("✨ Perso is ready! (Type 'exit' or 'quit' to end)\n");

    let stdin: io::Stdin = io::stdin();
    let mut stdout: io::Stdout = io::stdout();

    loop {
        print!("👤 You: ");
        stdout.flush()?;

        let mut input: String = String::new();
        stdin
            .read_line(&mut input)
            .context("Failed to read user input")?;

        let query: &str = input.trim();

        match query {
            "exit" | "quit" => {
                println!("👋 Goodbye!");
                break;
            }
            "" => continue,
            _ => {
                println!("🤖 Perso thinking...");
                match agent.prompt(query).await {
                    Ok(response) => println!("🤖 Perso: {}\n", response),
                    Err(e) => eprintln!("❌ Error: {}\n", e),
                }
            }
        }
    }

    Ok(())
}
