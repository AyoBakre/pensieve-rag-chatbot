import os
import streamlit as st
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
import uuid
import time
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Pensieve", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the UI
def apply_custom_css():
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 15px;
        min-height: 60px;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .st-emotion-cache-16txtl3 h2 {
        font-size: 1.8rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        max-width: 90%;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-top-right-radius: 0.25rem;
        margin-left: auto;
    }
    .chat-message.assistant {
        background-color: #343741;
        border-top-left-radius: 0.25rem;
    }
    .chat-message .content {
        display: inline-block;
    }
    .source-tag {
        font-size: 0.8rem;
        font-weight: bold;
        color: #00abff;
        margin-bottom: 0.5rem;
    }
    .relevance-high {
        color: #00cc66;
        font-weight: bold;
    }
    .relevance-medium {
        color: #ffcc00;
        font-weight: bold;
    }
    .relevance-low {
        color: #ff6666;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Pinecone client
def init_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
    
    # Create Pinecone client
    pc = pinecone.Pinecone(api_key=api_key)
    
    # List available indexes
    available_indexes = [index.name for index in pc.list_indexes()]
    
    return pc, available_indexes

# Initialize OpenAI client
def get_openai_client():
    # Using the correct API key format as provided
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
    return OpenAI(api_key=openai_api_key)

# Refine the user query to be more specific for better search results
def refine_query(client, query, chat_history):
    # Only refine if it seems like a vague query
    if len(query.split()) >= 4:
        return query
    
    # Get recent chat context
    recent_context = ""
    if chat_history and len(chat_history) > 0:
        recent_messages = chat_history[-3:]  # Get last 3 messages
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            recent_context += f"{role}: {msg['content']}\n"
    
    prompt = f"""
You are an AI tasked with improving search queries for a document retrieval system.
The user has entered a query that might be too vague or short to get good search results.
Your task is to refine it into a more specific, detailed query based on the conversation context.

Original query: "{query}"

Recent conversation context:
{recent_context}

Please rewrite the query to be more specific and detailed. Keep it concise but include relevant keywords.
If the query seems complete and specific already, or is a simple greeting, return it unchanged.
DO NOT add any explanation, ONLY return the refined query text.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You refine search queries to be more specific based on conversation context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        refined_query = response.choices[0].message.content
        # Strip any quotes from the response
        refined_query = refined_query.strip('"\'')
        
        if refined_query != query:
            st.session_state.latest_refinement = f"Query refined: '{query}' → '{refined_query}'"
        
        return refined_query
    except Exception as e:
        # If refinement fails, just return the original query
        return query

# Create embeddings using OpenAI
def get_embeddings(text, client):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Using text-embedding-3-small for better quality
    )
    return response.data[0].embedding

# Query Pinecone and get context
def query_pinecone(index, query_embedding, top_k=5):
    # Query the index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract the contexts from the results
    contexts = []
    for match in results.matches:
        if match.metadata and "text" in match.metadata:
            contexts.append({
                "text": match.metadata["text"],
                "score": match.score,
                "source": index.describe_index_stats().name if hasattr(index, 'describe_index_stats') else "Unknown",
                "id": match.id if hasattr(match, 'id') else None
            })
        elif match.metadata and "content" in match.metadata:
            contexts.append({
                "text": match.metadata["content"],
                "score": match.score,
                "source": index.describe_index_stats().name if hasattr(index, 'describe_index_stats') else "Unknown",
                "id": match.id if hasattr(match, 'id') else None
            })
    
    return contexts

# Query all Pinecone indexes
def query_all_indexes(pc, index_names, query_embedding, top_k_per_index=3):
    all_contexts = []
    sources_queried = {}
    
    # Query each index
    for index_name in index_names:
        try:
            index = pc.Index(index_name)
            contexts = query_pinecone(index, query_embedding, top_k_per_index)
            all_contexts.extend(contexts)
            sources_queried[index_name] = len(contexts) > 0  # Track which sources actually returned results
        except Exception as e:
            st.sidebar.warning(f"Failed to query index {index_name}: {str(e)}")
            sources_queried[index_name] = False
    
    # Sort all contexts by score (highest first)
    all_contexts.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top results across all indexes (limit to reasonable number to avoid token limits)
    return all_contexts[:min(len(all_contexts), top_k_per_index * 3)], sources_queried

# Evaluate relevance of each context to the query
def evaluate_context_relevance(client, query, contexts):
    if not contexts:
        return []
    
    # Prepare concise versions of contexts for evaluation
    context_texts = [ctx["text"][:200] + "..." for ctx in contexts]  # Truncate for token efficiency
    
    eval_prompt = f"""
For each document snippet below, evaluate its relevance to the query: "{query}"
Rate each as "HIGH" (very relevant), "MEDIUM" (somewhat relevant), or "LOW" (barely relevant).

"""
    
    for i, text in enumerate(context_texts):
        eval_prompt += f"Document {i+1}: {text}\n\n"
    
    eval_prompt += "Return ONLY a JSON array of relevance ratings in this exact format: [\"HIGH\", \"MEDIUM\", \"LOW\", ...] with no other text."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a document relevance evaluator. Only respond with the requested format."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        result = response.choices[0].message.content
        # Extract ratings from the response
        ratings = []
        try:
            # Try to extract a JSON array
            match = re.search(r'\[.*\]', result)
            if match:
                import json
                ratings_json = json.loads(match.group(0))
                ratings = ratings_json[:len(contexts)]  # Ensure we don't have more ratings than contexts
            else:
                # Fallback: extract HIGH, MEDIUM, LOW words
                ratings = re.findall(r'(HIGH|MEDIUM|LOW)', result)[:len(contexts)]
        except:
            # If extraction fails, default to MEDIUM
            ratings = ["MEDIUM"] * len(contexts)
        
        # Ensure we have a rating for each context
        while len(ratings) < len(contexts):
            ratings.append("MEDIUM")
        
        # Add ratings to contexts
        for i, ctx in enumerate(contexts):
            ctx["relevance"] = ratings[i] if i < len(ratings) else "MEDIUM"
        
        return contexts
    except Exception as e:
        # If evaluation fails, return contexts without relevance ratings
        for ctx in contexts:
            ctx["relevance"] = "MEDIUM"  # Default rating
        return contexts

# Choose appropriate model based on context complexity and query
def choose_model(query, contexts, history_length):
    # Use a more capable model for complex responses
    if any(ctx["relevance"] == "LOW" for ctx in contexts) and len(contexts) > 2:
        return "gpt-4o"  # More capable model for complex responses with varied relevance
    
    # Use a more capable model for longer conversations
    if history_length > 5:
        return "gpt-4o-mini"  # More capable model for deeper conversation context
    
    # Default to the standard model
    return "gpt-4o-mini"

# Format context with colored relevance indicators
def format_context_with_relevance(ctx):
    relevance = ctx.get("relevance", "MEDIUM")
    source = ctx.get("source", "Unknown")
    score = ctx.get("score", 0)
    
    relevance_display = {
        "HIGH": "🟢 HIGH RELEVANCE",
        "MEDIUM": "🟡 MEDIUM RELEVANCE",
        "LOW": "🔴 LOW RELEVANCE"
    }.get(relevance, "MEDIUM")
    
    return f"SOURCE: {source}\nRELEVANCE: {relevance_display}\nSCORE: {score:.2f}\nCONTENT: {ctx['text']}"

# Generate response using OpenAI
def generate_response(client, query, contexts, chat_history):
    # Check if we have any contexts
    if not contexts:
        return "I couldn't find any relevant information in your documents to answer this question. Could you try rephrasing your question or asking about something else?"
    
    # Format contexts with source information and relevance
    formatted_contexts = []
    sources_used = set()
    
    for i, ctx in enumerate(contexts):
        source = ctx.get('source', "Unknown")
        if source:  # Only add non-empty sources
            sources_used.add(source)
        formatted_contexts.append(format_context_with_relevance(ctx))
    
    context_text = "\n\n".join(formatted_contexts)
    
    # Format chat history for context
    history_text = ""
    if chat_history:
        history_entries = []
        for msg in chat_history[-6:]:  # Include up to last 6 messages
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_entries.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_entries)
    
    # Get current date and time for temporal context
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Choose model based on context complexity
    model = choose_model(query, contexts, len(chat_history))
    
    # Combine all contexts with the query
    prompt = f"""
You are Pensieve, an intelligent AI assistant that helps users find and understand information in their document collections.
Your purpose is to provide accurate, helpful, and concise answers based ONLY on the provided context.

Current date and time: {current_datetime}

CHAT HISTORY:
{history_text}

DOCUMENT CONTEXT (ordered by relevance):
{context_text}

USER QUESTION: {query}

Instructions:
1. Always cite your sources clearly - begin each main point with the source it came from.
2. Focus most on the HIGH RELEVANCE documents, and be more cautious with LOW RELEVANCE documents.
3. If different documents contain contradictory information, explicitly mention this discrepancy.
4. If the context contains information to answer the question, provide a clear and concise answer.
5. If answering a greeting or simple question, respond in a friendly way.
6. If you're unsure or the context doesn't contain relevant information, acknowledge this and suggest how they might rephrase their question.
7. Format your response in a readable way with paragraphs for different points.
8. If citing numbers, dates, or specific facts, always mention which specific source they came from.
9. When appropriate, offer 1-2 concrete suggestions for follow-up questions the user might want to ask.
10. DO NOT mention the relevance ratings in your response - they are for your judgment only.

RESPONSE:
"""
    
    # Generate response
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are Pensieve, a knowledgeable assistant with access to specific document collections. Always cite your sources clearly for each piece of information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # Balanced temperature for natural yet factual responses
    )
    
    response_text = response.choices[0].message.content
    
    # Post-process to ensure sources are mentioned
    has_mentioned_sources = False
    if sources_used:
        for source in sources_used:
            if source and source in response_text:
                has_mentioned_sources = True
                break
        
        if not has_mentioned_sources:
            sources_list = ", ".join([f"'{s}'" for s in sources_used if s])
            response_text = f"Based on information from {sources_list}:\n\n{response_text}"
        
        # Add a footer with metadata about the response
        sources_list = ", ".join([f"'{s}'" for s in sources_used if s])
        high_relevance_count = sum(1 for ctx in contexts if ctx.get("relevance") == "HIGH")
        
        metadata = []
        if sources_list:
            metadata.append(f"Sources: {sources_list}")
        
        metadata.append(f"Model: {model}")
        
        if high_relevance_count > 0:
            metadata.append(f"High relevance documents: {high_relevance_count}")
            
        if hasattr(st.session_state, 'latest_refinement'):
            metadata.append(st.session_state.latest_refinement)
            
        response_text += f"\n\n*{' | '.join(metadata)}*"
    
    return response_text

# Main app
def main():
    # Apply custom CSS
    apply_custom_css()
    
    st.title("🧠 Pensieve")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
    
    if "pinecone_client" not in st.session_state:
        st.session_state.pinecone_client = None
        
    if "available_indexes" not in st.session_state:
        st.session_state.available_indexes = []
    
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = get_openai_client()
    
    if "sources_queried" not in st.session_state:
        st.session_state.sources_queried = {}
    
    if "latest_refinement" not in st.session_state:
        st.session_state.latest_refinement = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Advanced settings expander
        with st.expander("Advanced Settings", expanded=False):
            st.session_state.top_k = st.slider("Results per index", min_value=1, max_value=10, value=3)
            st.session_state.auto_refine = st.checkbox("Auto-refine queries", value=True, help="Automatically enhance brief queries for better results")
        
        # Initialize Pinecone
        try:
            if st.session_state.pinecone_client is None:
                with st.spinner("Connecting to Pinecone..."):
                    st.session_state.pinecone_client, st.session_state.available_indexes = init_pinecone()
                
            if st.session_state.available_indexes:
                st.success(f"Connected to {len(st.session_state.available_indexes)} indexes")
                
                # Simple list of indexes with status indicators
                st.subheader("Available Document Collections")
                for idx, index_name in enumerate(sorted(st.session_state.available_indexes)):
                    status = "🔎" if st.session_state.sources_queried.get(index_name, False) else "⚪"
                    st.markdown(f"{status} **{idx+1}.** {index_name}")
                
                # Single row with two buttons
                col1, col2 = st.columns(2)
                with col1:
                    # Option to clear chat history
                    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear())
                with col2:
                    # Option to start a new chat - this is the only New Chat button
                    if st.button("New Chat"):
                        st.session_state.chat_id = str(uuid.uuid4())
                        st.session_state.messages = []
                
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.info("Please check your Pinecone API key.")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            role_class = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Check if connected to Pinecone
    if not st.session_state.available_indexes:
        st.info("Waiting for Pinecone connection...")
        return
    
    # User input - larger text area
    user_query = st.chat_input("Ask a question about your documents...", key="chat_input")
    
    if user_query and st.session_state.pinecone_client and st.session_state.available_indexes:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Set processing start time
                start_time = time.time()
                
                # Refine the query if needed and enabled
                if st.session_state.get("auto_refine", True) and len(user_query.split()) < 4:
                    refined_query = refine_query(
                        st.session_state.openai_client,
                        user_query,
                        st.session_state.messages[:-1]  # Exclude current query
                    )
                else:
                    refined_query = user_query
                
                # Get embeddings for the refined query
                query_embedding = get_embeddings(refined_query, st.session_state.openai_client)
                
                # Query all Pinecone indexes to get relevant contexts
                contexts, sources_queried = query_all_indexes(
                    st.session_state.pinecone_client,
                    st.session_state.available_indexes,
                    query_embedding,
                    top_k_per_index=st.session_state.get("top_k", 3)
                )
                
                # Update sources queried status
                st.session_state.sources_queried = sources_queried
                
                # Evaluate relevance of each context
                contexts_with_relevance = evaluate_context_relevance(
                    st.session_state.openai_client,
                    refined_query,
                    contexts
                )
                
                # Generate response using the context and chat history
                answer = generate_response(
                    st.session_state.openai_client,
                    user_query,
                    contexts_with_relevance,
                    st.session_state.messages[:-1]  # Pass all messages except current user query
                )
                
                # Calculate response time
                response_time = time.time() - start_time
                if not answer.endswith("*"):
                    answer += f"\n\n*Response time: {response_time:.2f} seconds*"
                else:
                    # Insert response time before last * if there's already a metadata footer
                    answer = answer[:-1] + f" | Response time: {response_time:.2f} seconds*"
                
                # Update placeholder with response
                message_placeholder.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Clear refinement note after use
                st.session_state.latest_refinement = None
                
                # Rerun to update the UI
                st.rerun()
                
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 