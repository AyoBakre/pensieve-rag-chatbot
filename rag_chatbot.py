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

# Document collection descriptions for suggesting initial questions
COLLECTION_DESCRIPTIONS = {
    "monet-legal-documents": {
        "description": "Legal documents including loan agreements and contracts",
        "suggested_questions": [
            "What is the exact interest rate calculation method in the SixPoint-Monet loan agreement?",
            "How does the SixPoint loan agreement define Organizational Documents?",
            "How is principal repayment scheduled in the Monet loan documents?",
            "What specific security provisions protect lenders in the Monet agreements?"
        ]
    },
    "monet-emails": {
        "description": "Email communications related to Monet",
        "suggested_questions": [
            "What is the status of the DocuSign process for the Monet TS v11 document?",
            "When was the EXECUTION VERSION Term Sheet shared with Monet?",
            "What specific attachments were included in recent DocuSign communications?",
            "Who are the key recipients of the term sheet execution emails?"
        ]
    },
    "ic-memo-documents": {
        "description": "Investment committee memos and financial documents",
        "suggested_questions": [
            "What exact revenue is Monet projecting for Q4 2024 vs December 2025?",
            "How many active users does Monet expect to have by December 2025 according to the One Pager?",
            "What are the specific capital needs ($3.5M equity breakdown) in Monet's investment memo?",
            "What does the impact study conducted by 60 Decibels reveal about Monet users?"
        ]
    },
    "monet-slack": {
        "description": "Slack messages and communications",
        "suggested_questions": [
            "What specific concerns were raised in Slack about Monet's revenue projections?",
            "Which specific Slack channels discuss the Monet term sheet negotiations?",
            "What was the team's response to the $3.5M equity requirement in Slack?",
            "What operational updates about Monet were shared in recent Slack messages?"
        ]
    },
    "monet-lsa-grain": {
        "description": "Additional loan security agreement documents",
        "suggested_questions": [
            "What specific collateral is required for Monet's loans according to the LSA documents?",
            "How are defaults specifically defined and handled in Monet's loan security agreement?",
            "What are the exact covenant requirements in Monet's loan security agreement?",
            "What specific reporting requirements must Monet satisfy under the loan documents?"
        ]
    }
}

# Generate initial suggested questions based on available indexes
def generate_suggested_questions(available_indexes):
    suggested_questions = []
    
    # Add general questions that work for any document collection but still specific to Monet
    general_questions = [
        "What are the key components of Monet's loan and security agreements?",
        "How is Monet planning to expand to 3.5 million users by 2025?",
        "What is Monet's approach to providing zero-interest loans with 8.4% commission?"
    ]
    suggested_questions.extend(general_questions)
    
    # Add specific questions for available indexes
    for index_name in available_indexes:
        if index_name in COLLECTION_DESCRIPTIONS:
            collection_info = COLLECTION_DESCRIPTIONS[index_name]
            # Add 1-2 questions from each available collection
            specific_questions = collection_info["suggested_questions"][:2]
            suggested_questions.extend(specific_questions)
    
    # Limit to a reasonable number of suggestions
    return suggested_questions[:6]

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
    /* Styling for suggestion buttons */
    .stButton > button {
        width: 100%;
        text-align: left;
        background-color: #1e293b;
        color: white;
        border: 1px solid #334155;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.2s ease;
        margin-bottom: 0.75rem;
        min-height: 3rem;
        white-space: normal;
        word-wrap: break-word;
    }
    .stButton > button:hover {
        background-color: #334155;
        border-color: #475569;
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    /* Style for follow-up question buttons - make them more distinctive */
    [data-testid*="follow"] button {
        background-color: #153e75;
        border-color: #2563eb;
        border-width: 1px;
        position: relative;
        padding-left: 2rem;
    }
    [data-testid*="follow"] button:hover {
        background-color: #1e4d8d;
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1), 0 2px 4px -1px rgba(59, 130, 246, 0.06);
    }
    [data-testid*="follow"] button::before {
        content: "↪️";
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1rem;
    }
    /* Style for source citations */
    strong em, em strong {
        color: #00abff;
        font-style: normal;
        font-weight: 600;
    }
    /* Citation highlight */
    .citation {
        background-color: rgba(0, 171, 255, 0.1);
        border-left: 3px solid #00abff;
        padding-left: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Pinecone client
def init_pinecone():
    # Get API key from environment variables or Streamlit secrets
    api_key = os.environ.get("PINECONE_API_KEY")
    if api_key is None and hasattr(st, 'secrets') and 'PINECONE_API_KEY' in st.secrets:
        api_key = st.secrets['PINECONE_API_KEY']
    
    # Create Pinecone client
    pc = pinecone.Pinecone(api_key=api_key)
    
    # List available indexes
    available_indexes = [index.name for index in pc.list_indexes()]
    
    return pc, available_indexes

# Initialize OpenAI client
def get_openai_client():
    # Get API key from environment variables or Streamlit secrets
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']
    
    return OpenAI(api_key=openai_api_key)

# Refine the user query to be more specific for better search results
def refine_query(client, query, chat_history):
    # Skip refinement for longer queries that are already specific
    if len(query.split()) >= 4:
        return query, None
    
    # Get recent chat context
    recent_context = ""
    if chat_history and len(chat_history) > 0:
        recent_messages = chat_history[-4:]  # Get last 4 messages for more context
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            recent_context += f"{role}: {msg['content']}\n"
    
    prompt = f"""
You are an AI tasked with improving search queries for a document retrieval system.
The user has entered a query that might be too vague, short, or ambiguous.

TASK:
1. Analyze the original query in context of the conversation
2. Create a more specific, detailed query that will yield better search results
3. Identify key entities, concepts, and relationships that should be emphasized
4. Add context-specific terms from the conversation history
5. Format: Return two elements - the refined query text followed by a brief reasoning

Original query: "{query}"

Recent conversation context:
{recent_context}

DO NOT include any explanation in your refined query. Only return:
REFINED_QUERY: [your improved query]
REASONING: [brief explanation of changes]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use faster model for better performance
            messages=[
                {"role": "system", "content": "You refine search queries to be more specific based on conversation context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        result = response.choices[0].message.content
        
        # Extract refined query and reasoning
        refined_query = query  # Default to original
        reasoning = None
        
        query_match = re.search(r'REFINED_QUERY:\s*(.*?)(?:\n|$)', result)
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:\n|$)', result, re.DOTALL)
        
        if query_match:
            refined_query = query_match.group(1).strip('" ')
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        if refined_query != query:
            return refined_query, reasoning
        else:
            return query, None
            
    except Exception as e:
        # If refinement fails, just return the original query
        return query, None

# Create embeddings using OpenAI
def get_embeddings(text, client):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Using text-embedding-3-small which outputs 1536 dimensions
    )
    return response.data[0].embedding

# Query Pinecone and get context with hybrid retrieval
def query_pinecone(index, query_embedding, query_text, top_k=5):
    """Query using both vector similarity and optional keyword boost"""
    try:
        # Vector similarity query (dense retrieval)
        vector_results = index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Retrieve more candidates for hybrid ranking
            include_metadata=True
        )
        
        # Extract the contexts from the vector results
        contexts = []
        for match in vector_results.matches:
            if match.metadata:
                text = match.metadata.get("text") or match.metadata.get("content", "")
                if text:
                    contexts.append({
                        "text": text,
                        "score": match.score,
                        "source": index.describe_index_stats().name,
                        "id": match.id,
                        "retrieval_method": "vector",
                        "keywords": []  # Will be populated if keyword match
                    })
        
        # Extract important keywords from the query for sparse retrieval
        important_keywords = extract_keywords(query_text)
        
        # If we have keywords, re-rank results based on keyword presence
        if important_keywords:
            for ctx in contexts:
                # Count keyword matches
                keyword_matches = []
                for keyword in important_keywords:
                    if keyword.lower() in ctx["text"].lower():
                        keyword_matches.append(keyword)
                
                # Store matched keywords
                ctx["keywords"] = keyword_matches
                
                # Boost score based on keyword matches (hybrid scoring)
                keyword_boost = min(len(keyword_matches) * 0.05, 0.3)  # Cap boost at 0.3
                ctx["original_score"] = ctx["score"]
                ctx["score"] = ctx["score"] + keyword_boost
                
                # Track if this was boosted by keywords
                if keyword_boost > 0:
                    ctx["retrieval_method"] = "hybrid"
        
        # Sort by the (potentially boosted) score
        contexts.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k after hybrid ranking
        return contexts[:top_k]
    
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# Extract important keywords from query
def extract_keywords(query_text, client=None):
    """Extract key search terms for sparse retrieval"""
    # Simple extraction for short queries
    if len(query_text.split()) <= 3:
        return [word for word in query_text.split() if len(word) > 3]
    
    # For longer queries, try NLP-based extraction if OpenAI client available
    if client:
        try:
            prompt = f"""
Extract the 3-5 most important keywords or phrases from this search query.
Return ONLY a comma-separated list of keywords, no explanations.

QUERY: {query_text}
KEYWORDS:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract key search terms from queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            keywords = response.choices[0].message.content.strip().split(',')
            return [k.strip() for k in keywords if k.strip()]
        except:
            # Fall back to simple extraction on error
            pass
    
    # Simple fallback - words longer than 3 chars, exclude stopwords
    stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'are', 'any', 'from', 'what', 'how', 'when', 'where', 'who', 'why'}
    return [word for word in query_text.split() if len(word) > 3 and word.lower() not in stopwords]

# Query all Pinecone indexes with hybrid retrieval
def query_all_indexes(pc, index_names, query_embedding, query_text, openai_client=None, top_k_per_index=3):
    all_contexts = []
    sources_queried = {}
    
    # Extract keywords for sparse retrieval component
    keywords = extract_keywords(query_text, openai_client)
    
    # Query each index
    for index_name in index_names:
        try:
            index = pc.Index(index_name)
            contexts = query_pinecone(index, query_embedding, query_text, top_k_per_index)
            all_contexts.extend(contexts)
            sources_queried[index_name] = len(contexts) > 0  # Track which sources actually returned results
        except Exception as e:
            st.sidebar.warning(f"Failed to query index {index_name}: {str(e)}")
            sources_queried[index_name] = False
    
    # Sort all contexts by score (highest first)
    all_contexts.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top results across all indexes (limit to reasonable number to avoid token limits)
    return all_contexts[:min(len(all_contexts), top_k_per_index * 3)], sources_queried

# Choose appropriate model based on context complexity and query
def choose_model(query, contexts, history_length):
    # Use a more capable model for complex responses that require reasoning
    complex_query_indicators = ["why", "how", "explain", "compare", "analyze", "evaluate", "difference", "relationship"]
    query_lowercase = query.lower()
    query_is_complex = any(indicator in query_lowercase for indicator in complex_query_indicators)
    
    # Use the most capable model only for very complex scenarios
    if query_is_complex and any(ctx["relevance"] == "LOW" for ctx in contexts) and len(contexts) > 3 and history_length > 5:
        return "gpt-4o"  # Use the most powerful model only for very complex reasoning with long context
    
    # Default to the faster model for most queries
    return "gpt-4o-mini"

# Format context with colored relevance indicators
def format_context_with_relevance(ctx):
    relevance = ctx.get("relevance", "MEDIUM")
    source = ctx.get("source", "Unknown")
    score = ctx.get("score", 0)
    confidence = ctx.get("confidence", "Medium")
    
    relevance_display = {
        "HIGH": "🟢 HIGH RELEVANCE",
        "MEDIUM": "🟡 MEDIUM RELEVANCE",
        "LOW": "🔴 LOW RELEVANCE"
    }.get(relevance, "MEDIUM")
    
    metadata = []
    if "created_date" in ctx:
        metadata.append(f"DATE: {ctx['created_date']}")
    if "author" in ctx:
        metadata.append(f"AUTHOR: {ctx['author']}")
        
    metadata_str = "\n".join(metadata) + "\n" if metadata else ""
    
    return f"SOURCE: {source}\nRELEVANCE: {relevance_display}\nSCORE: {score:.2f}\nCONFIDENCE: {confidence}\n{metadata_str}CONTENT: {ctx['text']}"

# Enhanced context relevance evaluation with reasoning
def evaluate_context_relevance(client, query, contexts):
    if not contexts:
        return []
    
    # Prepare concise versions of contexts for evaluation
    context_texts = [ctx["text"][:500] + "..." for ctx in contexts]  # Use more text for better judgment
    
    eval_prompt = f"""
You are evaluating the relevance of document snippets to a user query.
For each document snippet, you will:
1. Analyze how directly it answers the query
2. Identify key information that relates to the query
3. Consider contradictions or supporting evidence between documents
4. Rate each as "HIGH" (directly answers query), "MEDIUM" (contains related information), or "LOW" (tangentially related)
5. Assign a confidence level of "High", "Medium", or "Low" to your rating

QUERY: "{query}"

"""
    
    for i, text in enumerate(context_texts):
        eval_prompt += f"DOCUMENT {i+1}: {text}\n\n"
    
    eval_prompt += """For each document, provide a JSON object with this exact format:
[
  {"relevance": "HIGH/MEDIUM/LOW", "confidence": "High/Medium/Low", "reasoning": "Brief explanation"},
  ...
]
Respond ONLY with the JSON array, no other text."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use faster model for evaluation
            messages=[
                {"role": "system", "content": "You are a document relevance evaluator with expertise in information retrieval. Be precise in your evaluations."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1,
            max_tokens=2000  # Allow more tokens for detailed analysis
        )
        
        result = response.choices[0].message.content
        # Extract ratings from the response
        try:
            # Try to extract a JSON array
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                import json
                evaluations = json.loads(match.group(0))
                
                # Add detailed evaluations to contexts
                for i, ctx in enumerate(contexts):
                    if i < len(evaluations):
                        ctx["relevance"] = evaluations[i].get("relevance", "MEDIUM")
                        ctx["confidence"] = evaluations[i].get("confidence", "Medium")
                        ctx["reasoning"] = evaluations[i].get("reasoning", "")
                    else:
                        ctx["relevance"] = "MEDIUM"
                        ctx["confidence"] = "Low"
                        ctx["reasoning"] = "Not evaluated"
            else:
                # Fallback: extract HIGH, MEDIUM, LOW words
                ratings = re.findall(r'(HIGH|MEDIUM|LOW)', result)[:len(contexts)]
                for i, ctx in enumerate(contexts):
                    ctx["relevance"] = ratings[i] if i < len(ratings) else "MEDIUM"
                    ctx["confidence"] = "Low"
                    ctx["reasoning"] = "Basic extraction"
        except Exception as e:
            # If JSON extraction fails, use regex fallback
            ratings = re.findall(r'(HIGH|MEDIUM|LOW)', result)[:len(contexts)]
            for i, ctx in enumerate(contexts):
                ctx["relevance"] = ratings[i] if i < len(ratings) else "MEDIUM"
                ctx["confidence"] = "Low"
                ctx["reasoning"] = f"Extraction fallback: {str(e)[:50]}"
        
        # Re-rank contexts based on relevance
        high_relevance = [ctx for ctx in contexts if ctx["relevance"] == "HIGH"]
        medium_relevance = [ctx for ctx in contexts if ctx["relevance"] == "MEDIUM"]
        low_relevance = [ctx for ctx in contexts if ctx["relevance"] == "LOW"]
        
        # Sort by score within each relevance group
        high_relevance.sort(key=lambda x: x.get("score", 0), reverse=True)
        medium_relevance.sort(key=lambda x: x.get("score", 0), reverse=True)
        low_relevance.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Combine re-ranked contexts
        reranked_contexts = high_relevance + medium_relevance + low_relevance
        
        return reranked_contexts
    except Exception as e:
        # If evaluation fails, return contexts without relevance ratings
        for ctx in contexts:
            ctx["relevance"] = "MEDIUM"  # Default rating
            ctx["confidence"] = "Low"
            ctx["reasoning"] = f"Evaluation error: {str(e)[:50]}"
        return contexts

# Extract follow-up questions from the response
def extract_follow_up_questions(client, query, response, contexts):
    try:
        # Get sources for context
        sources = set()
        document_types = set()
        document_names = set()
        
        for ctx in contexts:
            if "source" in ctx and ctx["source"]:
                sources.add(ctx["source"])
            
            # Extract additional metadata if available
            if "metadata" in ctx:
                meta = ctx["metadata"]
            else:
                meta = ctx
                
            if "document_type" in meta:
                document_types.add(meta["document_type"])
            if "document_name" in meta:
                document_names.add(meta["document_name"])
        
        # Extract missing or incomplete information
        prompt = f"""
Given the user's question and the assistant's response about Monet (a financial technology company), generate 3 highly specific follow-up questions.

DOCUMENT CONTEXT:
- Sources: {', '.join(sources)}
- Document types: {', '.join(document_types)}
- Document names: {', '.join(document_names)}

These follow-up questions should:
1. Address specific aspects not fully covered in the response
2. Probe deeper into the financial details, business model, or legal arrangements
3. Reference specific documents, agreements, or numbers mentioned in the response
4. Be naturally phrased and conversational
5. Be relevant to Monet's business (financial technology, loans, credit facilities, etc.)

USER QUESTION: {query}

ASSISTANT RESPONSE: {response}

Provide EXACTLY 3 follow-up questions as a JSON array:
["Question 1", "Question 2", "Question 3"]

Make each question specific and actionable - avoid generic questions that don't relate directly to Monet's business or the documents mentioned.
"""
        
        # Generate follow-up questions using a fast model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate targeted follow-up questions about Monet, a fintech company focused on loans and financial inclusion."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        result = response.choices[0].message.content
        
        # Extract questions from JSON array
        import re
        import json
        
        # Find JSON array in response
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            questions = json.loads(match.group(0))
            st.session_state.follow_up_questions = questions[:3]  # Limit to 3 questions
        else:
            # Fallback parsing if JSON extraction fails
            questions = re.findall(r'"([^"]+)"', result)
            if questions:
                st.session_state.follow_up_questions = questions[:3]
            else:
                # Simple line-by-line extraction as last resort
                questions = [line.strip() for line in result.split('\n') if '?' in line]
                st.session_state.follow_up_questions = questions[:3]
    
    except Exception as e:
        print(f"Error generating follow-up questions: {str(e)}")
        st.session_state.follow_up_questions = []

# Generate response using OpenAI with enhanced prompting
def generate_response(client, query, contexts, chat_history, refined_reasoning=None):
    # Check if we have any contexts
    if not contexts:
        return "I couldn't find any relevant information in your documents to answer this question. Could you try rephrasing your question or asking about something else?"
    
    # Format contexts with source information and relevance
    formatted_contexts = []
    sources_used = set()
    
    # Group contexts by source for better organization
    contexts_by_source = {}
    for ctx in contexts:
        source = ctx.get('source', "Unknown")
        if source not in contexts_by_source:
            contexts_by_source[source] = []
        contexts_by_source[source].append(ctx)
        if source:  # Only add non-empty sources
            sources_used.add(source)
    
    # Format contexts grouped by source, prioritizing HIGH relevance
    for source, ctx_group in contexts_by_source.items():
        # Sort by relevance and then by score
        ctx_group.sort(key=lambda x: (
            0 if x.get("relevance") == "HIGH" else 1 if x.get("relevance") == "MEDIUM" else 2,
            -x.get("score", 0)
        ))
        
        for ctx in ctx_group:
            formatted_contexts.append(format_context_with_relevance(ctx))
    
    context_text = "\n\n" + "\n\n".join(formatted_contexts)
    
    # Extract key reasoning from contexts (if available)
    key_insights = []
    for ctx in contexts:
        if ctx.get("relevance") == "HIGH" and ctx.get("reasoning"):
            key_insights.append(f"- {ctx.get('reasoning')}")
    
    insights_text = ""
    if key_insights:
        insights_text = "\nKEY DOCUMENT INSIGHTS:\n" + "\n".join(key_insights[:3])
    
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
    
    # Add context about query refinement if available
    refinement_note = ""
    if refined_reasoning:
        refinement_note = f"\nQUERY REFINEMENT NOTE: {refined_reasoning}"
    
    # Combine all contexts with the query
    prompt = f"""
You are Pensieve, an intelligent AI assistant that helps users find and understand information in their document collections.
Your purpose is to provide accurate, helpful, and concise answers based ONLY on the provided context.

Current date and time: {current_datetime}

CHAT HISTORY:
{history_text}

USER QUESTION: {query}{refinement_note}{insights_text}

DOCUMENT CONTEXT (ordered by relevance):
{context_text}

RESPONSE GUIDELINES:
1. ANALYZE each document for relevance to the question.
2. PRIORITIZE information from HIGH RELEVANCE documents.
3. SYNTHESIZE information across multiple sources when appropriate.
4. CITE your sources clearly - format citations as "According to [Source Name]" or "[Source Name] states that..."
5. HIGHLIGHT contradictions between documents explicitly.
6. ACKNOWLEDGE uncertainty when the context is incomplete or unclear.
7. STRUCTURE your response with clear paragraphs and transitions.
8. INCLUDE specific facts, figures, and quotes with their sources.
9. OMIT the relevance ratings - they are for your judgment only.

RESPONSE STRUCTURE:
1. Brief introductory statement addressing the question
2. Main points with clear source citations in the format "According to [Source Name]..."
3. Summary that synthesizes the information
"""
    
    # Create system message
    system_message = {
        "role": "system", 
        "content": "You are Pensieve, a knowledgeable assistant with access to document collections. You prioritize accuracy and clarity, providing well-structured responses that ALWAYS cite sources by name using the format 'According to [Source Name]...' at least once per paragraph. You focus on directly answering the user's question using only information from the provided documents."
    }
    
    # Create user message
    user_message = {
        "role": "user", 
        "content": prompt
    }
    
    # Return the stream object for the caller to process
    return {
        "stream": client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=1000,  # Allow longer responses for complex questions
            stream=True  # Enable streaming
        ),
        "sources_used": sources_used,
        "model": model,
        "high_relevance_count": sum(1 for ctx in contexts if ctx.get("relevance") == "HIGH"),
        "refined_reasoning": refined_reasoning
    }

# Main app with added user feedback
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
    
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {}
        
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
        
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
        
    if "follow_up_questions" not in st.session_state:
        st.session_state.follow_up_questions = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Advanced settings expander
        with st.expander("Advanced Settings", expanded=False):
            st.session_state.top_k = st.slider("Results per index", min_value=1, max_value=10, value=3)
            st.session_state.auto_refine = st.checkbox("Auto-refine queries", value=True, help="Automatically enhance brief queries for better results")
            st.session_state.model_selector = st.selectbox(
                "Response model",
                ["Auto select", "gpt-4o-mini", "gpt-4o"], 
                index=0,
                help="Choose which model to use for responses"
            )
            st.session_state.show_debug = st.checkbox("Show debug info", value=False, help="Show detailed information about retrieval and relevance")
        
        # Initialize Pinecone
        try:
            if st.session_state.pinecone_client is None:
                with st.spinner("Connecting to Pinecone..."):
                    st.session_state.pinecone_client, st.session_state.available_indexes = init_pinecone()
                    # Generate suggested questions based on available indexes
                    st.session_state.suggested_questions = generate_suggested_questions(st.session_state.available_indexes)
                
            if st.session_state.available_indexes:
                st.success(f"Connected to {len(st.session_state.available_indexes)} indexes")
                
                # Simple list of indexes with status indicators
                st.subheader("Available Document Collections")
                for idx, index_name in enumerate(sorted(st.session_state.available_indexes)):
                    status = "🔎" if st.session_state.sources_queried.get(index_name, False) else "⚪"
                    st.markdown(f"{status} **{idx+1}.** {index_name}")
                
                # Chat controls
                st.subheader("Chat Controls")
                
                cols = st.columns(2)
                with cols[0]:
                    # Option to clear chat history
                    if st.button("Clear Chat", key="clear_chat"):
                        st.session_state.messages = []
                        st.session_state.follow_up_questions = []
                        st.success("Chat history cleared!")
                
                with cols[1]:
                    # Option to start a new chat
                    if st.button("New Chat", key="new_chat"):
                        st.session_state.chat_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.follow_up_questions = []
                        st.success("Started a new chat!")
                
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
                
                # Add feedback buttons for assistant messages
                if message["role"] == "assistant" and i > 0:
                    message_id = str(i)
                    
                    # Check if we already have feedback for this message
                    if message_id in st.session_state.feedback_data:
                        feedback = st.session_state.feedback_data[message_id]
                        if feedback == "positive":
                            st.success("You marked this response as helpful")
                        elif feedback == "negative":
                            st.error("You marked this response as not helpful")
                    else:
                        cols = st.columns([1, 1, 8])
                        with cols[0]:
                            if st.button("👍 Helpful", key=f"pos_{i}"):
                                st.session_state.feedback_data[message_id] = "positive"
                                # Use experimental_rerun instead of return
                                st.rerun()
                        with cols[1]:
                            if st.button("👎 Not Helpful", key=f"neg_{i}"):
                                st.session_state.feedback_data[message_id] = "negative"
                                # Use experimental_rerun instead of return
                                st.rerun()
    
    # Check if connected to Pinecone
    if not st.session_state.available_indexes:
        st.info("Waiting for Pinecone connection...")
        return
    
    # Display suggested questions if no messages yet
    if len(st.session_state.messages) == 0 and st.session_state.suggested_questions:
        st.subheader("I can help you find information in your documents. Try asking:")
        
        # Display suggested questions as clickable buttons in rows of 2
        for i in range(0, len(st.session_state.suggested_questions), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(st.session_state.suggested_questions):
                    question = st.session_state.suggested_questions[i + j]
                    if cols[j].button(f"🔍 {question}", key=f"sugg_{i+j}"):
                        # Use this question as the user input
                        user_query = question
                        # Process the query (copied from below)
                        st.session_state.messages.append({"role": "user", "content": user_query})
                        with st.chat_message("user"):
                            st.write(user_query)
                        process_query(user_query)
                        st.rerun()  # Use rerun instead of return
    
    # Display follow-up questions if available
    if len(st.session_state.messages) > 0 and st.session_state.follow_up_questions:
        # Create a visually distinct container for follow-up questions
        follow_up_container = st.container()
        with follow_up_container:
            st.markdown("""
            <div style="margin-top: 20px; margin-bottom: 15px; padding: 10px; border-radius: 8px; background-color: #1b2838; border-left: 4px solid #5ba3e0;">
                <h4 style="color: #5ba3e0; margin-bottom: 10px; font-weight: 600;">Follow-up questions you might ask:</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a more visually distinct layout for follow-up questions
            for i, question in enumerate(st.session_state.follow_up_questions):
                # Create a unique key for each button to avoid conflicts
                button_key = f"follow_{i}_{hash(question)}"
                
                # Wrap the button in a container for better styling
                if st.button(question, key=button_key, help="Click to ask this follow-up question"):
                    # Use this question as the user input
                    user_query = question
                    # Process the query
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    with st.chat_message("user"):
                        st.write(user_query)
                    process_query(user_query)
                    st.rerun()  # Use rerun instead of return
    
    # User input - larger text area
    user_query = st.chat_input("Ask a question about your documents...", key="chat_input")
    
    if user_query and st.session_state.pinecone_client and st.session_state.available_indexes:
        # Clear any previous follow-up questions when a new direct query is received
        st.session_state.follow_up_questions = []
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process the query
        process_query(user_query)

# Process a user query (extracted from main for reuse with suggested questions)
def process_query(user_query):
    # Clear any previous follow-up questions when processing a new query
    st.session_state.follow_up_questions = []
    
    # Generate and display response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_placeholder.markdown("Thinking...")
        
        try:
            # Set processing start time
            start_time = time.time()
            
            # Refine the query if needed and enabled
            if st.session_state.get("auto_refine", True) and len(user_query.split()) < 4:
                refined_query, refined_reasoning = refine_query(
                    st.session_state.openai_client,
                    user_query,
                    st.session_state.messages[:-1]  # Exclude current query
                )
            else:
                refined_query = user_query
                refined_reasoning = None
            
            # Get embeddings for the refined query
            query_embedding = get_embeddings(refined_query, st.session_state.openai_client)
            
            # Query all Pinecone indexes to get relevant contexts
            contexts, sources_queried = query_all_indexes(
                st.session_state.pinecone_client,
                st.session_state.available_indexes,
                query_embedding,
                refined_query,
                st.session_state.openai_client,
                top_k_per_index=st.session_state.get("top_k", 3)
            )
            
            # Update sources queried status
            st.session_state.sources_queried = sources_queried
            
            # Show debug information if enabled
            if st.session_state.show_debug:
                debug_info = f"Query: '{user_query}'"
                if refined_query != user_query:
                    debug_info += f" → Refined: '{refined_query}'"
                
                debug_info += f"\nRetrieved {len(contexts)} contexts from {len([s for s,v in sources_queried.items() if v])} sources"
                
                # Count retrieval methods
                vector_count = len([c for c in contexts if c.get("retrieval_method") == "vector"])
                hybrid_count = len([c for c in contexts if c.get("retrieval_method") == "hybrid"])
                
                debug_info += f"\nRetrieval methods: {vector_count} vector, {hybrid_count} hybrid"
                
                with st.expander("Debug Information", expanded=False):
                    st.write(debug_info)
                    
                    # Show keyword matches if any
                    keyword_matches = []
                    for ctx in contexts:
                        if ctx.get("keywords"):
                            keyword_matches.extend(ctx.get("keywords"))
                    
                    if keyword_matches:
                        st.write(f"Keyword matches: {', '.join(set(keyword_matches))}")
            
            # Evaluate relevance of each context
            contexts_with_relevance = evaluate_context_relevance(
                st.session_state.openai_client,
                refined_query,
                contexts
            )
            
            # Force model selection if user has chosen a specific model
            model_selection = st.session_state.get("model_selector", "Auto select")
            if model_selection != "Auto select":
                # Override the choose_model function by directly setting the model
                chosen_model = model_selection
            else:
                # Use the normal model selection logic
                chosen_model = choose_model(
                    refined_query, 
                    contexts_with_relevance,
                    len(st.session_state.messages)
                )
            
            # Get streaming response data
            response_data = generate_response(
                st.session_state.openai_client,
                user_query,
                contexts_with_relevance,
                st.session_state.messages[:-1],  # Pass all messages except current user query
                refined_reasoning
            )
            
            # Extract metadata for the footer
            sources_used = response_data["sources_used"]
            model = response_data["model"]
            high_relevance_count = response_data["high_relevance_count"]
            refined_reasoning = response_data["refined_reasoning"]
            
            # Display the streaming response
            for chunk in response_data["stream"]:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # Update the message placeholder with the accumulated response
                    message_placeholder.markdown(full_response + "▌")
                    
            # Calculate response time
            response_time = time.time() - start_time
            
            # Create metadata footer
            metadata = []
            if sources_used:
                sources_list = ", ".join([f"'{s}'" for s in sources_used if s])
                if sources_list:
                    metadata.append(f"Sources: {sources_list}")
            
            metadata.append(f"Model: {model}")
            
            if high_relevance_count > 0:
                metadata.append(f"High relevance documents: {high_relevance_count}")
                
            if refined_reasoning:
                metadata.append(f"Query refinement: {refined_reasoning[:50]}...")
            
            metadata.append(f"Response time: {response_time:.2f} seconds")
            
            # Add metadata footer to the response
            full_response += f"\n\n*{' | '.join(metadata)}*"
            
            # Final update without the cursor character
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Generate follow-up questions
            extract_follow_up_questions(st.session_state.openai_client, user_query, full_response, contexts_with_relevance)
            
            # Clear refinement note after use
            st.session_state.latest_refinement = None
            
        except Exception as e:
            message_placeholder.markdown(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 