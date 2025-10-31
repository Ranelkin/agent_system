import streamlit as st
import sys, os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.infrastructure.llm.graph import main_graph
from src.infrastructure.mcp.client import get_mcp_manager, shutdown_mcp
from src.shared.log_config import setup_logging

load_dotenv()
logger = setup_logging("streamlit_app")

def extract_clean_text(content):
    """Extract clean text from various message formats"""
    if isinstance(content, str):
        if "structuredContent=" in content:
            import re
            match = re.search(r"'result':\s*'([^']+)'", content)
            if match:
                return match.group(1)
        return content
    return str(content)

st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("AI Assistant")

with st.sidebar:
    st.markdown(
        """
        - **Web searches**  
        - **Unit tests**  
        - **Documentation**  
        - **Investment analysis**  
        """
    )
    if st.button("End Session"):
        shutdown_mcp()
        st.success("Session cleaned up")
        st.rerun()

@st.cache_resource
def init_mcp():
    manager = get_mcp_manager()
    logger.info("MCP manager initialized")
    return manager

init_mcp()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("What would you like to do?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_ph = st.empty()
        input_data = {"messages": [{"role": "user", "content": prompt}]}
        final_answer = ""
        
        try:
            step_container = st.container()
            
            step_names = {
                "router": "🔀 Routing",
                "extract_ticker": "🎯 Extracting Ticker",
                "fetch_data": "📡 Fetching Market Data",
                "fundamental": "📊 Fundamental Analysis",
                "technical": "📈 Technical Analysis",
                "mediator": "🎓 Final Recommendation",
                "search_agent": "🔍 Web Search",
                "test_agent": "🧪 Creating Tests",
                "comment_agent": "📝 Documentation",
            }
            
            last_state = None
            
            for event in main_graph.stream(
                input_data,
                stream_mode="updates",
                config={"recursion_limit": 50},
            ):
                for node_name, node_output in event.items():
                    friendly_name = step_names.get(node_name, f"⚙️ {node_name}")
                    
                    with step_container.expander(friendly_name, expanded=False):
                        if "messages" in node_output:
                            msgs = node_output["messages"]
                            for msg in msgs:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "")
                                    txt = msg.get("content", "")
                                else:
                                    role = getattr(msg, "type", "assistant")
                                    txt = getattr(msg, "content", "")
                                
                                clean_txt = extract_clean_text(txt)
                                if clean_txt and clean_txt.strip():
                                    st.markdown(clean_txt)
                    
                    last_state = node_output
            
            if last_state and "messages" in last_state:
                final_msgs = last_state["messages"]
                
                for msg in reversed(final_msgs):
                    if isinstance(msg, dict):
                        role = msg.get("role", "").lower()
                        txt = msg.get("content", "")
                    else:
                        role = getattr(msg, "type", "").lower()
                        txt = getattr(msg, "content", "")
                    
                    if role not in {"user", "human"} and txt:
                        final_answer = extract_clean_text(txt)
                        break
            
            if final_answer:
                answer_ph.markdown(final_answer)
            else:
                answer_ph.error("No final answer – check the steps above.")
        
        except Exception as e:
            logger.error(f"Graph error: {e}", exc_info=True)
            answer_ph.error(f"Error: {e}")
        
        final_display = final_answer or "*(no answer)*"
        st.session_state.messages.append({"role": "assistant", "content": final_display})

def _shutdown():
    try:
        shutdown_mcp()
        logger.info("MCP shutdown")
    except Exception as ex:
        logger.error(f"Shutdown error: {ex}")

import atexit
atexit.register(_shutdown)